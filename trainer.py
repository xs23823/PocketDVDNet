import os
import time
import csv
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tensorboardX import SummaryWriter  # COMMENTED OUT - protobuf issues on blue pebble but uncomment for local
import numpy as np

from torch.optim.lr_scheduler import StepLR
from utils.lr_scheduler import MultiStepRestartLR, CosineAnnealingRestartLR

from dataloaders.fastdvdnet.utils import *
from dataloaders.noise import NoiseModel

from train_method.obproxsg import OBProxSG
class DistillationLoss(nn.Module):
    """balanced loss between teacher and ground truth"""
    def __init__(self, alpha=0.5, loss_type='mse'):
        super().__init__()
        self.alpha = alpha
        
        if loss_type == 'mse':
            self.base_loss = nn.MSELoss(reduction='sum')
        elif loss_type == 'l1':
            self.base_loss = nn.L1Loss(reduction='sum')
        elif loss_type == 'charbonnier':
            self.base_loss = CharbonnierLoss(reduction='sum')
        else:
            self.base_loss = nn.MSELoss(reduction='sum')
    
    def forward(self, student_output, teacher_target, gt_target):
        """
        args:
            student_output: model predictions [B, C, H, W]
            teacher_target: teacher outputs [B, C, H, W]
            gt_target: ground truth [B, C, H, W]
        returns:
            balanced loss
        """
        teacher_loss = self.base_loss(student_output, teacher_target)
        gt_loss = self.base_loss(student_output, gt_target)
        
        # alpha controls teacher weight, (1-alpha) is gt weight
        total_loss = self.alpha * teacher_loss + (1 - self.alpha) * gt_loss
        return total_loss



class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon=1e-3, reduction='sum'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
    
    def forward(self, output, gt):
        loss = torch.sqrt((output - gt) ** 2 + self.epsilon ** 2)
        if self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'mean':
            return loss.mean()
        return loss


class Trainer:
    def __init__(self, args, model, prefetcher, val_loader, start_epoch=0):

        self.args = args
        self.model = model
        self.prefetcher = prefetcher
        self.val_loader = val_loader

        self.device = next(model.parameters()).device
        self.epoch = start_epoch
        self.iteration = 0
        self.best_psnr = 0.0
        
        self.use_obproxsg = getattr(self.args, 'use_obproxsg', False)
        if self.use_obproxsg:
            self.best_sparsity = 0.0
            self.best_sparsity_info = None
            self.is_best_sparsity = False
            self.sparsity_threshold = getattr(self.args, 'sparsity_threshold', 0.0)
        self.use_distillation = getattr(self.args, 'use_distillation', False)
        if self.use_distillation:
            self.distill_alpha = getattr(self.args, 'distill_alpha', 0.5)
            print(f"distillation enabled with alpha={self.distill_alpha}")


        
        self.is_best = False

        # setup optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()
        self.scaler = torch.amp.GradScaler("cuda")
        self.load()

        self.current_lr = self.get_lr()
        
        # setup loss function
        self.setup_criterion()
       

        # tensorboard
        self.summary = {}
        self.log_dir = os.path.join(args.out_dir, "logs")
        self.writer = SummaryWriter(self.log_dir)
        self.writer = None  # disable tensorboard

        # noise model
        self.noise_fn = NoiseModel("./dataloaders/predicted_labels.csv")

    def setup_criterion(self):
        """setup loss function based on config"""
        loss_type = getattr(self.args, 'loss_type', 'mse').lower()
        
        if self.use_distillation:
            self.criterion = DistillationLoss(
                alpha=self.distill_alpha,
                loss_type=loss_type
            )
            print(f"using distillation loss with {loss_type} base")
        else:
            if loss_type == 'mse':
                self.criterion = nn.MSELoss(reduction="sum")
            elif loss_type == 'l1':
                self.criterion = nn.L1Loss(reduction="sum")
            elif loss_type == 'charbonnier':
                self.criterion = CharbonnierLoss(reduction="sum")
            else:
                print(f"unknown loss type {loss_type}, defaulting to mse")
                self.criterion = nn.MSELoss(reduction="sum")
            
            print(f"using {loss_type} loss")

    def setup_optimizers(self):
        """Set up optimizers"""
        backbone_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                backbone_params.append(param)
            else:
                print(f"Params {name} will not be optimized.")

        # check if OBProxSG for sparsity training
        if self.use_obproxsg:
            print("Using OBProxSG optimizer")
            
            self.optimizer = OBProxSG(
                backbone_params,
                lr=self.args.lr,
                lambda_reg=getattr(self.args, 'lambda_reg', 1e-5),
                epochSize=getattr(self.args, 'epochSize', 1000),
                Np=getattr(self.args, 'Np', 2),
                No=getattr(self.args, 'No', 'inf'),
                eps=getattr(self.args, 'eps', 1e-4),
                weight_decay=getattr(self.args, 'weight_decay', 0),
                lambda_warmup_steps=getattr(self.args, 'lambda_warmup_steps', 10000)
            )
        else:
            # Original Adam optimizer
            optim_params = [
                {"params": backbone_params, "lr": self.args.lr},
            ]
            self.optimizer = torch.optim.Adam(optim_params)

    def setup_schedulers(self):
        """Set up schedulers."""
        scheduler_opt = self.args.scheduler
        scheduler_type = scheduler_opt["name"]

        if scheduler_type in ["MultiStepLR", "MultiStepRestartLR"]:
            self.scheduler = MultiStepRestartLR(
                self.optimizer,
                milestones=scheduler_opt["milestones"],
                gamma=scheduler_opt["gamma"],
            )

        elif scheduler_type == "CosineAnnealingRestartLR":
            self.scheduler = CosineAnnealingRestartLR(
                self.optimizer,
                periods=scheduler_opt["periods"],
                restart_weights=scheduler_opt["restart_weights"],
                eta_min=scheduler_opt["eta_min"],
            )

        else:
            print(
                f"Scheduler {scheduler_type} is not recognized. Defaulting to StepLR."
            )
            self.scheduler = StepLR(
                self.optimizer,
                step_size=scheduler_opt["step_size"],
                gamma=scheduler_opt["gamma"],
            )

    def update_learning_rate(self):
        """Update learning rate."""
        self.scheduler.step()

    def get_lr(self):
        """Get current learning rate."""
        return self.optimizer.param_groups[0]["lr"]
    
    def get_model_sparsity(self):
        """Return model sparsity percentage and raw counts."""
        total = 0
        zeros = 0
        for p in self.model.parameters():
            if p.requires_grad:
                total += p.numel()
                zeros += (p == 0).sum().item()
        sparsity = 100.0 * zeros / total if total > 0 else 0
        return sparsity, zeros, total

    def add_summary(self, writer, name, val):
        """Add tensorboard summary """
        if name not in self.summary:
            self.summary[name] = 0
        self.summary[name] += val
        n = self.args.log_freq
        if writer is not None and self.iteration % n == 0:
            writer.add_scalar(name, self.summary[name] / n, self.iteration)
            self.summary[name] = 0
        pass

    def load(self):
        """Load Model and Optimizer."""

        if os.path.isfile(os.path.join(self.args.out_dir, "latest.pt")):
            ckpt = torch.load(
                os.path.join(self.args.out_dir, "latest.pt"), map_location=self.device, weights_only=False
            )
            latest_epoch = ckpt.get("epoch", None)
            self.best_psnr = 0.0
        else:
            latest_epoch = None
            self.best_psnr = 0.0

        if latest_epoch is not None:
            model_path = os.path.join(self.args.out_dir, f"latest.pt")
            print(f"Loading model epoch {latest_epoch} from {model_path}")

            ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
            self.epoch = ckpt["epoch"]
            self.iteration = ckpt["iteration"]
            self.model.load_state_dict(ckpt["model_state"])
            self.optimizer.load_state_dict(ckpt["optim_state"])
            self.scheduler.load_state_dict(ckpt["sched_state"])

               
            if getattr(self.args, 'reset_lr', False):
                print(f"Resetting learning rate from {self.optimizer.param_groups[0]['lr']:.2e} to {self.args.lr:.2e}")
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.args.lr
                self.setup_schedulers()

        else:
            print("Training from scratch!")
            self.best_psnr = 0.0

    def save(self):
        """Save latest checkpoint and, if flagged, update best checkpoint."""
        
        # base checkpoint dict
        ckpt = {
            "epoch": self.epoch,
            "iteration": self.iteration,
            "model_state": self.model.state_dict(),
            "optim_state": self.optimizer.state_dict(),
            "sched_state": self.scheduler.state_dict(),
            "best_psnr": self.best_psnr,
        }
        
        # add sparsity if using OBProxSG
        if self.use_obproxsg:
            sparsity, zeros, total = self.get_model_sparsity()
            ckpt["sparsity"] = sparsity
            ckpt["best_sparsity"] = self.best_sparsity

        # always overwrite latest.pt
        latest_path = os.path.join(self.args.out_dir, "latest.pt")
        torch.save(ckpt, latest_path)

        #  best PSNR (both)
        if self.is_best:
            if self.use_obproxsg:
                # if OBProxSG, save as best_psnr.pt to distinguish from sparsity 
                best_path = os.path.join(self.args.out_dir, "best_psnr.pt")
            else:
                best_path = os.path.join(self.args.out_dir, "best.pt")
            torch.save(ckpt, best_path)

        #  best sparsity if OBProxSG
        if self.use_obproxsg and self.is_best_sparsity:
            best_sparsity_path = os.path.join(self.args.out_dir, "best_sparsity.pt")
            torch.save(ckpt, best_sparsity_path)
            sparsity, zeros, total = self.get_model_sparsity()
            self.best_sparsity_info = (zeros, total)

    def train(self):

        pbar = range(int(self.args.iterations))
        pbar = tqdm(pbar, initial=self.iteration, dynamic_ncols=True, smoothing=0.01)

        while self.iteration < self.args.iterations:
            self.epoch += 1
            self.prefetcher.reset()
            self.train_epoch(pbar)
            self.validate()
            self.save()

        pbar.close()
        tqdm.write("\nTraining complete.")
        
        #  sparsity info if OBProxSG
        if self.use_obproxsg:
            if self.best_sparsity_info:
                zeros, total = self.best_sparsity_info
                sparsity = 100.0 * zeros / total if total > 0 else 0
                tqdm.write(f"Best sparsity model: {zeros} out of {total} parameters zeroed ({sparsity:.2f}%)")
            else:
                # Fallback 
                sparsity, zeros, total = self.get_model_sparsity()
                tqdm.write(f"Final model sparsity: {zeros} out of {total} parameters zeroed ({sparsity:.2f}%)")

    def train_epoch(self, pbar):
        """process input and calculate loss every training epoch"""
        self.model.train()
        
        while True:
            batch = self.prefetcher.next()
            if batch is None:
                break
            
            self.iteration += 1
            
            if self.use_distillation:
                # distillation mode - no noise addition
                noisy_input = batch['noisy_input'].to(self.device)
                teacher_target = batch['teacher_target'].to(self.device)
                gt_target = batch['gt_target'].to(self.device)
                
                B, CF, H, W = noisy_input.shape
                
                # create noise map for model (using fixed std as pacnet was trained on sigma 30)
                noise_std = 30.0 / 255.0  # pacnet default
                noise_map = torch.full((B, 21, H, W), noise_std, device=self.device)
                
                # forward pass
                self.optimizer.zero_grad()
                
                with torch.amp.autocast("cuda"):
                    output = self.model(noisy_input, noise_map)
                    loss = self.criterion(output, teacher_target, gt_target) / B
                    
            else:
                # standard training mode with noise addition
                img_train, gt_train = batch
                img_train, gt_train = img_train.to(self.device), gt_train.to(self.device)
                
                # existing noise addition code...
                B, _, H, W = img_train.size()
                stdn = torch.empty((B, 1, 1, 1)).to(self.device).uniform_(
                    self.args.noise_ival[0], to=self.args.noise_ival[1]
                )
                noise = torch.zeros_like(img_train).to(self.device)
                noise = torch.normal(mean=noise, std=stdn.expand_as(noise))
                
                imgn_train = img_train + noise
                noise_map = stdn.expand((B, 21, H, W)).to(self.device)
                
                self.optimizer.zero_grad()
                
                with torch.amp.autocast("cuda"):
                    output = self.model(imgn_train, noise_map)
                    loss = self.criterion(output, gt_train) / (B * 2)
            
            # backpropagation
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # logging
            pbar.update(1)
            if self.iteration % 30 == 0:
                self.model.apply(svd_orthogonalization)
                self.current_lr = self.get_lr()
                pbar.set_description(f"LR: {self.current_lr} Loss: {loss.item():.3f}")
        
        torch.cuda.empty_cache()
        self.update_learning_rate()

    def validate(self):
        """validate the model - always adds noise for validation"""
        self.model.eval()
        
        psnr_val = 0
        t1 = time.time()
        
        with torch.no_grad():
            for seq_val in tqdm(self.val_loader, leave=False, desc="Validating"):
                # validation always adds noise since val sequences are clean
                noise = torch.FloatTensor(seq_val.size()).normal_(
                    mean=0, std=self.args.val_noiseL
                )
                seqn_val = seq_val + noise
                seqn_val = seqn_val.to(self.device, non_blocking=True)
                
                # prepare noise standard deviation tensor
                sigma_noise = torch.tensor(
                    [self.args.val_noiseL], dtype=torch.float32, device=self.device
                )
                
                # perform denoising
                out_val = denoise_seq_fastdvdnet(
                    seq=seqn_val[0], noise_std=sigma_noise, model_temporal=self.model
                )
                
                # calculate psnr
                psnr_val += batch_psnr(out_val.cpu(), seq_val.squeeze_(), 1.0)
            
            psnr_val /= len(self.val_loader)
            t2 = time.time()
        
        # check if current model is the best
        if psnr_val > self.best_psnr:
            self.best_psnr = psnr_val
            self.is_best = True
        else:
            self.is_best = False
        
        # sparsity tracking if using obproxsg
        if self.use_obproxsg:
            current_sparsity, zeros, total = self.get_model_sparsity()
            
            if psnr_val >= self.sparsity_threshold and current_sparsity > self.best_sparsity:
                self.best_sparsity = current_sparsity
                self.is_best_sparsity = True
                print(f"new best sparsity: {current_sparsity:.2f}% (psnr: {psnr_val:.4f})")
            else:
                self.is_best_sparsity = False
        
        # status message
        if self.use_obproxsg:
            current_sparsity, _, _ = self.get_model_sparsity()
            status_msg = f"\n[epoch {self.epoch}] psnr_val: {psnr_val:.4f} (best: {self.is_best}), sparsity: {current_sparsity:.1f}%, on {t2-t1:.2f} sec"
        else:
            status_msg = f"\n[epoch {self.epoch}] psnr_val: {psnr_val:.4f} (best: {self.is_best}), on {t2-t1:.2f} sec"
        
        tqdm.write(status_msg)