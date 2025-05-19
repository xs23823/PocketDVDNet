import os
import time
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.tensorboard import SummaryWriter

from torch.optim.lr_scheduler import StepLR
from utils.lr_scheduler import MultiStepRestartLR, CosineAnnealingRestartLR

from dataloaders.fastdvdnet.utils import *


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
        self.is_best = False

        # setup optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()
        self.scaler = torch.amp.GradScaler("cuda")
        self.load()

        self.current_lr = self.get_lr()
        self.criterion = nn.MSELoss(reduction="sum")

        # Set up tensorboard
        self.summary = {}
        self.log_dir = os.path.join(args.out_dir, "logs")
        self.writer = SummaryWriter(self.log_dir)

    def setup_optimizers(self):
        """Set up optimizers."""
        backbone_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                backbone_params.append(param)
            else:
                print(f"Params {name} will not be optimized.")

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
            # Default to StepLR
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

    def add_summary(self, writer, name, val):
        """Add tensorboard summary."""
        if name not in self.summary:
            self.summary[name] = 0
        self.summary[name] += val
        n = self.args.log_freq
        if writer is not None and self.iteration % n == 0:
            writer.add_scalar(name, self.summary[name] / n, self.iteration)
            self.summary[name] = 0

    def load(self):
        """Load Model and Optimizer."""

        if os.path.isfile(os.path.join(self.args.out_dir, "latest.pt")):
            ckpt = torch.load(
                os.path.join(self.args.out_dir, "latest.pt"), map_location=self.device
            )
            latest_epoch = ckpt.get("epoch", None)
            self.best_psnr = ckpt.get("best_psnr", 0.0)
        else:
            latest_epoch = None
            self.best_psnr = 0.0

        if latest_epoch is not None:
            model_path = os.path.join(self.args.out_dir, f"latest.pt")
            print(f"Loading model epoch {latest_epoch} from {model_path}")

            ckpt = torch.load(model_path, map_location=self.device)
            self.epoch = ckpt["epoch"]
            self.iteration = ckpt["iteration"]
            self.model.load_state_dict(ckpt["model_state"])
            self.optimizer.load_state_dict(ckpt["optim_state"])
            self.scheduler.load_state_dict(ckpt["sched_state"])

        else:
            print("Training from scratch!")
            self.best_psnr = 0.0

    def save(self):
        """Save latest checkpoint and, if flagged, update best checkpoint."""
        # bundle everything into one dict
        ckpt = {
            "epoch": self.epoch,
            "iteration": self.iteration,
            "model_state": self.model.state_dict(),
            "optim_state": self.optimizer.state_dict(),
            "sched_state": self.scheduler.state_dict(),
            "best_psnr": self.best_psnr,
        }

        # always overwrite latest.pt
        latest_path = os.path.join(self.args.out_dir, "latest.pt")
        torch.save(ckpt, latest_path)
        print(f"Saved latest checkpoint to {latest_path}")

        # if this is the best so far, also overwrite best.pt
        if self.is_best:
            best_path = os.path.join(self.args.out_dir, "best.pt")
            torch.save(ckpt, best_path)
            print(f"Saved best checkpoint to   {best_path}")

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

    def train_epoch(self, pbar):
        """
        Process input and calculate loss every training epoch
        """

        self.model.train()

        while True:
            batch = self.prefetcher.next()
            if batch is None:
                break

            # Unpack the batch and move to device
            img_train, gt_train = batch
            img_train, gt_train = img_train.to(self.device), gt_train.to(self.device)

            self.iteration += 1

            B, _, H, W = img_train.size()
            stdn = (
                torch.empty((B, 1, 1, 1))
                .to(self.device)
                .uniform_(self.args.noise_ival[0], to=self.args.noise_ival[1])
            )
            # draw noise samples from std dev tensor
            noise = torch.zeros_like(img_train).to(self.device)
            noise = torch.normal(mean=noise, std=stdn.expand_as(noise))

            # define noisy inputs
            imgn_train = img_train + noise
            noise_map = stdn.expand((B, 1, H, W)).to(
                self.device
            )

            self.optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                output = self.model(imgn_train, noise_map)
                loss = self.criterion(output, gt_train) / (B * 2)

            # Backpropagation
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.add_summary(self.writer, "loss", loss.item())

            # Console logs
            pbar.update(1)
            if self.iteration % 10 == 0:
                self.model.apply(svd_orthogonalization)
                self.current_lr = self.get_lr()
                pbar.set_description((f"LR: {self.current_lr} Loss: {loss.item():.3f}"))

        # Clean up
        torch.cuda.empty_cache()
        self.update_learning_rate()

        # # Log training images
        # img = torchvision.utils.make_grid(
        #     img_train.view(-1, 3, H, W),
        #     nrow=8, normalize=True, scale_each=True
        # )
        # self.writer.add_image('Training patches', img, self.epoch)

        tqdm.write(
            "Train Epoch: {} [{}/{} ({:.0f}%)] - [Loss: {:.6f}]".format(
                self.epoch,
                self.iteration,
                self.args.iterations,
                100.0,  # * batch_idx / len(train_loader),
                loss.item(),
            )
        )

    def validate(self):
        """Validate the model and log images to TensorBoard."""

        self.model.eval()

        psnr_val = 0
        t1 = time.time()

        with torch.no_grad():
            for seq_val in tqdm(self.val_loader, leave=False, desc="Validating"):
                # Add noise to the validation sequence
                noise = torch.FloatTensor(seq_val.size()).normal_(
                    mean=0, std=self.args.val_noiseL
                )
                seqn_val = seq_val + noise
                seqn_val = seqn_val.to(self.device, non_blocking=True)

                # Prepare noise standard deviation tensor
                sigma_noise = torch.tensor(
                    [self.args.val_noiseL], dtype=torch.float32, device=self.device
                )

                # Perform denoising
                out_val = denoise_seq_fastdvdnet(
                    seq=seqn_val[0], noise_std=sigma_noise, model_temporal=self.model
                )

                # Calculate PSNR
                psnr_val += batch_psnr(out_val.cpu(), seq_val.squeeze_(), 1.0)

            psnr_val /= len(self.val_loader)
            t2 = time.time()

        # Log PSNR and learning rate
        self.writer.add_scalar("PSNR on validation data", psnr_val, self.epoch)
        self.writer.add_scalar("Learning rate", self.current_lr, self.epoch)

        # Check if current model is the best
        if psnr_val > self.best_psnr:
            self.best_psnr = psnr_val
            self.is_best = True
            print(f"\nNew best model found! PSNR: {psnr_val:.4f}")
        else:
            self.is_best = False

        # Log validation images
        idx = 0

        # Log clean and noisy validation images
        img = torchvision.utils.make_grid(
            seq_val.data[idx].clamp(0.0, 1.0), nrow=2, normalize=False, scale_each=False
        )
        imgn = torchvision.utils.make_grid(
            seqn_val.data[0][idx].clamp(0.0, 1.0),
            nrow=2,
            normalize=False,
            scale_each=False,
        )
        self.writer.add_image("Clean validation image {}".format(idx), img, self.epoch)
        self.writer.add_image("Noisy validation image {}".format(idx), imgn, self.epoch)

        # Log reconstructed validation results
        irecon = torchvision.utils.make_grid(
            out_val.data[idx].clamp(0.0, 1.0), nrow=2, normalize=False, scale_each=False
        )
        self.writer.add_image(
            "Reconstructed validation image {}".format(idx), irecon, self.epoch
        )

        print(f"\n[epoch {self.epoch}] PSNR_val: {psnr_val:.4f}, on {t2-t1:.2f} sec")
