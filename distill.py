import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import yaml
import time
import sys
from pathlib import Path
from tqdm.auto import tqdm
import torchvision

# Import necessary modules
from dataloaders.noise import NoiseModel
from models import PocketDVDnet, PocketDVDnet7
from studfastdvdnet.models.shiftnetwrapper import ShiftNet
from dataloaders.fastdvdnet import ValDataset, DVDDataset, Sampler
from utils.prefetcher import PrefetchDataLoader, CPUPrefetcher
from trainer import CharbonnierLoss
from dataloaders.fastdvdnet.utils import batch_psnr, denoise_seq_fastdvdnet
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
from torch.optim.lr_scheduler import StepLR
from utils.lr_scheduler import MultiStepRestartLR, CosineAnnealingRestartLR

# TensorBoard imports
try:
    from tensorboardX import SummaryWriter
except ImportError:
    from torch.utils.tensorboard import SummaryWriter

class OnFlyDistillationTrainer:
    """trains pocketdvdnet with shiftnet as teacher"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # create output dir if needed
        self.out_dir = Path(args.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        # set sequence params
        self.sequence_length = getattr(args, 'sequence_length', 7)
        self.ctrl_fr_idx = self.sequence_length // 2
        print(f"Using sequence length: {self.sequence_length}, center frame index: {self.ctrl_fr_idx}")
        
        # Initialize models
        self._init_models()
        
        # Initialize datasets and loaders
        self._init_datasets()
        
        # Initialize training components
        self._init_training()
        
        # Track training state
        self.epoch = 0
        self.iteration = 0
        self.best_psnr = 0.0
        self.is_best = False
        
        # Initialize TensorBoard logging
        self.log_dir = os.path.join(args.out_dir, "logs")
        self.writer = SummaryWriter(self.log_dir)
        print(f"TensorBoard logs will be saved to: {self.log_dir}")
        
        # Load checkpoint if exists
        self.load()

        self.noise_model = NoiseModel(dict_path="./dataloaders/predicted_labels.csv", num_frames=self.sequence_length)
        self.val_noise_model = NoiseModel(dict_path="./dataloaders/predicted_labels.csv", seed=1, num_frames=self.sequence_length)
    
    def _init_models(self):
        """init teacher and student models"""
        # setup teacher
        print("Initializing ShiftNet as teacher model...")
        self.teacher = ShiftNet().to(self.device)
        
        # Load ShiftNet checkpoint
        if not hasattr(self.args, 'teacher_checkpoint') or not os.path.exists(self.args.teacher_checkpoint):
            raise ValueError(f"Teacher checkpoint not found: {getattr(self.args, 'teacher_checkpoint', 'Not specified')}")
        
        self.teacher.load_checkpoint(self.args.teacher_checkpoint, self.device)
        print(f"Loaded teacher model from {self.args.teacher_checkpoint}")
        self.teacher.eval()  # Set teacher to evaluation mode
        
        # Freeze teacher parameters
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        # Initialize student model (PocketDVDnet)
        print("Initializing PocketDVDnet as student model...")
        if self.sequence_length == 7:
            print(f"Using PocketDVDnet7 model ({self.sequence_length}-frame)")
            self.student = PocketDVDnet7(num_input_frames=self.sequence_length).to(self.device)
        elif self.sequence_length == 5:
            print(f"Using PocketDVDnet model ({self.sequence_length}-frame)")
            self.student = PocketDVDnet(num_input_frames=self.sequence_length).to(self.device)
        else:
            raise ValueError(f"Unsupported sequence length: {self.sequence_length}. Must be 5 or 7.")
        
        # Optionally load pretrained student weights
        if hasattr(self.args, 'student_pretrained') and os.path.exists(self.args.student_pretrained):
            print(f"Loading pretrained student weights from {self.args.student_pretrained}")
            student_ckpt = torch.load(self.args.student_pretrained, map_location=self.device)
            if "model_state" in student_ckpt:
                self.student.load_state_dict(student_ckpt["model_state"])
            elif "state_dict" in student_ckpt:
                self.student.load_state_dict(student_ckpt["state_dict"])
            else:
                self.student.load_state_dict(student_ckpt)
    
    def _init_datasets(self):
        """setup training and val datasets"""
        print(f"Setting up training dataset from {self.args.trainset_dir}")
        train_dataset = DVDDataset(
            root_dir=self.args.trainset_dir,
            sequence_length=self.sequence_length,
            ctrl_fr_idx=self.ctrl_fr_idx,
            crop_size=self.args.patch_size,
        )
        
        # Setup validation dataset
        print(f"Setting up validation dataset from {self.args.valset_dir}")
        val_dataset = ValDataset(
            valsetdir=self.args.valset_dir, 
            gray_mode=False,
            num_input_frames=self.sequence_length
        )
        
        # Setup data loaders
        self.train_loader = PrefetchDataLoader(
            self.args.num_prefetch_queue,
            dataset=train_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=False,
            shuffle=True,
            drop_last=True,
        )
        
        self.val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=1,
            num_workers=2,
            pin_memory=True,
            shuffle=False,
        )
        
        self.prefetcher = CPUPrefetcher(self.train_loader)
    
    def _init_training(self):
        """init optimizer, scheduler, and loss"""
        self.optimizer = optim.Adam(self.student.parameters(), lr=self.args.lr)
        
        # Setup scheduler
        scheduler_config = getattr(self.args, 'scheduler', {'name': 'StepLR', 'step_size': 30, 'gamma': 0.6})
        scheduler_type = scheduler_config["name"]

        if scheduler_type in ["MultiStepLR", "MultiStepRestartLR"]:
            self.scheduler = MultiStepRestartLR(
                self.optimizer,
                milestones=scheduler_config["milestones"],
                gamma=scheduler_config["gamma"],
            )
        elif scheduler_type == "CosineAnnealingRestartLR":
            self.scheduler = CosineAnnealingRestartLR(
                self.optimizer,
                periods=scheduler_config["periods"],
                restart_weights=scheduler_config["restart_weights"],
                eta_min=scheduler_config["eta_min"],
            )
        else:
            print(f"Scheduler {scheduler_type} is not recognized. Defaulting to StepLR.")
            self.scheduler = StepLR(
                self.optimizer,
                step_size=scheduler_config.get("step_size", 30),
                gamma=scheduler_config.get("gamma", 0.6)
            )
        
        # Setup loss functions
        loss_type = getattr(self.args, 'loss_type', 'charbonnier')
        if loss_type == 'charbonnier':
            print("Using charbonnier loss for distillation")
            self.criterion = CharbonnierLoss(reduction='sum')
        elif loss_type == 'l1':
            print("Using L1 loss for distillation")
            self.criterion = nn.L1Loss(reduction='sum')
        else:
            print("Using MSE loss for distillation")
            self.criterion = nn.MSELoss(reduction='sum')
        
        # Setup mixed precision training
        self.scaler = torch.amp.GradScaler('cuda')
        
        # Distillation alpha (weight between teacher and GT targets)
        self.distill_alpha = getattr(self.args, 'distill_alpha', 0.5)
        print(f"Distillation alpha: {self.distill_alpha} (teacher: {self.distill_alpha}, GT: {1-self.distill_alpha})")
    
    def load(self):
        """Load Model and Optimizer."""
        if os.path.isfile(os.path.join(self.args.out_dir, "latest.pt")):
            latest_path = os.path.join(self.args.out_dir, "latest.pt")
            print(f"Loading checkpoint from {latest_path}")

            ckpt = torch.load(latest_path, map_location=self.device, weights_only=False)
            self.epoch = ckpt.get("epoch", 0)
            self.iteration = ckpt.get("iteration", 0)
            self.best_psnr = ckpt.get("best_psnr", 0.0)
            
            self.student.load_state_dict(ckpt["model_state"])
            self.optimizer.load_state_dict(ckpt["optim_state"])
            self.scheduler.load_state_dict(ckpt["sched_state"])
            
            print(f"Loaded checkpoint: epoch {self.epoch}, iteration {self.iteration}, best PSNR: {self.best_psnr:.4f}")
        else:
            print("Training from scratch!")
            self.best_psnr = 0.0

    def save(self):
        """Save latest checkpoint and, if flagged, update best checkpoint."""
        ckpt = {
            "epoch": self.epoch,
            "iteration": self.iteration,
            "model_state": self.student.state_dict(),
            "optim_state": self.optimizer.state_dict(),
            "sched_state": self.scheduler.state_dict(),
            "best_psnr": self.best_psnr,
        }

        # Save latest checkpoint
        latest_path = os.path.join(self.args.out_dir, "latest.pt")
        torch.save(ckpt, latest_path)

        # Save best checkpoint if this is the best model
        if self.is_best:
            best_path = os.path.join(self.args.out_dir, "best.pt")
            torch.save(ckpt, best_path)
            print(f"New best model saved with PSNR: {self.best_psnr:.4f}")

    def get_lr(self):
        """Get current learning rate."""
        return self.optimizer.param_groups[0]["lr"]

    def update_learning_rate(self):
        """Update learning rate."""
        self.scheduler.step()
    
    def calculate_psnr(self, output, target):
        """Calculate PSNR between output and target tensors"""
        mse = torch.mean((output - target) ** 2)
        if mse == 0:
            return float('inf')
        return -10 * torch.log10(mse)
    
    def log_images_to_tensorboard(self, clean_img, noisy_img, teacher_output, student_output, gt, iteration, prefix="Train"):
        """Log images to TensorBoard for visualization"""
        try:
            # Convert from [B, F*C, H, W] to [B, C, H, W] for center frame
            B, FC, H, W = clean_img.shape
            C = 3  # RGB
            F = FC // C
            center_idx = F // 2
            
            # Extract center frame from each sequence
            clean_center = clean_img.view(B, F, C, H, W)[:, center_idx]  # [B, C, H, W]
            noisy_center = noisy_img.view(B, F, C, H, W)[:, center_idx]  # [B, C, H, W]
            
            # Use first batch item for visualization
            clean_vis = torchvision.utils.make_grid(clean_center[0:1].cpu().clamp(0, 1), normalize=False)
            noisy_vis = torchvision.utils.make_grid(noisy_center[0:1].cpu().clamp(0, 1), normalize=False)
            teacher_vis = torchvision.utils.make_grid(teacher_output[0:1].cpu().clamp(0, 1), normalize=False)
            student_vis = torchvision.utils.make_grid(student_output[0:1].cpu().clamp(0, 1), normalize=False)
            
            # Log individual images
            self.writer.add_image(f'{prefix}/Clean_Input', clean_vis, iteration)
            self.writer.add_image(f'{prefix}/Noisy_Input', noisy_vis, iteration)
            self.writer.add_image(f'{prefix}/Teacher_Output', teacher_vis, iteration)
            self.writer.add_image(f'{prefix}/Student_Output', student_vis, iteration)
            
            # Create comparison grid
            comparison = torch.cat([noisy_vis, teacher_vis, student_vis, clean_vis], dim=2)
            self.writer.add_image(f'{prefix}/Comparison_Noisy_Teacher_Student_GT', comparison, iteration)
            
        except Exception as e:
            print(f"Warning: Failed to log images to TensorBoard: {e}")

    def train(self):
        """Main training loop"""
        print("Starting on-the-fly distillation training...")
        
        pbar = range(int(self.args.iterations))
        pbar = tqdm(pbar, initial=self.iteration, dynamic_ncols=True, smoothing=0.01, file=sys.stdout)

        while self.iteration < self.args.iterations:
            self.epoch += 1
            self.prefetcher.reset()
            self.train_epoch(pbar)
            self.validate()
            self.save()

        pbar.close()
        print("\nTraining complete!")
        
        # Close TensorBoard writer
        if self.writer is not None:
            self.writer.close()
            print(f"TensorBoard logs saved to: {self.log_dir}")
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'writer') and self.writer is not None:
            self.writer.close()
    
    def train_epoch(self, pbar):
        """Training epoch with on-the-fly teacher generation"""
        self.student.train()
        prev_batch = None
        
        while True:
            batch = self.prefetcher.next()
            if batch is None:
                break
                
            # Debug check for duplicate batches
            if prev_batch is not None:
                img_prev, _ = prev_batch
                img_curr, _ = batch
                if torch.equal(img_prev, img_curr):
                    print(f"Warning: Duplicate batch detected at iteration {self.iteration}")
            prev_batch = batch
            
            self.iteration += 1
            
            # get input and gt
            img_train, gt_train = batch


            img_train = img_train.to(self.device)
            gt_train = gt_train.to(self.device)

            # add noise
            B, _, H, W = img_train.size()
            # stdn = torch.empty((B, 1, 1, 1)).to(self.device).uniform_(
            #     self.args.noise_ival[0], to=self.args.noise_ival[1]
            # )
            # noise = torch.zeros_like(img_train).to(self.device)
            # noise = torch.normal(mean=noise, std=stdn.expand_as(noise))
            
            # imgn_train = img_train + noise
            # print("Clean input shape:", img_train.shape, "min:", img_train.min().item(), "max:", img_train.max().item())
            imgn_train, noise_map = self.noise_model(img_train)
            # print("Noisy input shape:", imgn_train.shape, "min:", imgn_train.min().item(), "max:", imgn_train.max().item())
            # print("Noise map shape:", noise_map.shape, "min:", noise_map.min().item(), "max:", noise_map.max().item())
            imgn_train = imgn_train.reshape(B, self.sequence_length * 3, H, W)
            
            # Prepare noise maps for both models
            # For student (PocketDVDnet) - expects [B, sequence_length*3, H, W] noise map
            # student_noise_map = stdn.expand((B, self.sequence_length * 3, H, W)).to(self.device)
            # student_noise_map = noise_map.repeat(1, 1, 3, 1, 1).view(B, self.sequence_length * 3, H, W).to(self.device)
            
            # # For teacher (ShiftNet) - expects [B, 1, H, W] noise map
            # teacher_noise_map = stdn.expand((B, 1, H, W)).to(self.device)
            teacher_noise_map = noise_map[:, 0, :, :, :].view(B, 1, H, W).to(self.device)
            
            # forward pass
            self.optimizer.zero_grad()
            
            with torch.amp.autocast("cuda"):
                # get teacher output (no gradients)
                with torch.no_grad():
                    teacher_output = self.teacher(imgn_train, teacher_noise_map)
                
                # get student output
                student_output = self.student(imgn_train)
                
                # calc losses
                teacher_loss = self.criterion(student_output, teacher_output) / (B * 2)
                gt_loss = self.criterion(student_output, gt_train) / (B * 2)
                
                # Combined loss with weighting
                loss = (self.distill_alpha * gt_loss) + teacher_loss
            
            # Backpropagation
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Calculate PSNR values for logging
            with torch.no_grad():
                teacher_psnr = self.calculate_psnr(teacher_output, gt_train)
                student_psnr = self.calculate_psnr(student_output, gt_train)
                teacher_vs_student_psnr = self.calculate_psnr(student_output, teacher_output)
            
            # TensorBoard logging
            if self.writer is not None:
                # Log scalar metrics
                self.writer.add_scalar('Train/Total_Loss', loss.item(), self.iteration)
                self.writer.add_scalar('Train/Teacher_Loss', teacher_loss.item(), self.iteration)
                self.writer.add_scalar('Train/GT_Loss', gt_loss.item(), self.iteration)
                self.writer.add_scalar('Train/Learning_Rate', self.get_lr(), self.iteration)
                self.writer.add_scalar('Train/Teacher_PSNR', teacher_psnr.item(), self.iteration)
                self.writer.add_scalar('Train/Student_PSNR', student_psnr.item(), self.iteration)
                self.writer.add_scalar('Train/Student_vs_Teacher_PSNR', teacher_vs_student_psnr.item(), self.iteration)
                
                # Log images every 100 iterations for first few epochs or every 1000 iterations later
                log_freq = 100 if self.epoch <= 5 else 1000
                if self.iteration % log_freq == 0:
                    # Note: Both teacher and student now see the SAME noisy images (imgn_train)
                    self.log_images_to_tensorboard(
                        img_train, imgn_train, teacher_output, student_output, gt_train, 
                        self.iteration, prefix="Train"
                    )
            
            # Update progress bar
            pbar.update(1)
            current_lr = self.get_lr()
            pbar.set_description(
                f"LR: {current_lr:.6f} Loss: {loss.item():.3f} "
                f"T-PSNR: {teacher_psnr.item():.2f} S-PSNR: {student_psnr.item():.2f}"
            )
            
            # Check if we've reached the iteration limit
            if self.iteration >= self.args.iterations:
                break
        
        torch.cuda.empty_cache()
        self.update_learning_rate()
    
    def validate(self):
        """Validate the student model on validation dataset"""
        self.student.eval()
        
        psnr_val = 0
        t1 = time.time()
        validation_logged = False
        
        with torch.no_grad():
            for i, seq_val in enumerate(tqdm(self.val_loader, leave=False, desc="Validating")):

                B, F, C, H, W = seq_val.size()

                seqn_val, noise_map = self.val_noise_model(seq_val.view(B, F * C, H, W).to(self.device))
                # Perform denoising
                out_val = denoise_seq_fastdvdnet(
                    seq=seqn_val[0], model_temporal=self.student
                )
                
                # Calculate PSNR
                psnr_batch = batch_psnr(out_val.cpu(), seq_val.squeeze_().cpu(), 1.0)
                psnr_val += psnr_batch
                
                # Log validation images for first batch only
                if i == 0 and not validation_logged and self.writer is not None:
                    # Get center frame for visualization
                    F, C, H, W = out_val.shape
                    center_idx = F // 2
                    
                    # Create visualization tensors
                    clean_center = seq_val.squeeze_()[center_idx:center_idx+1]  # [1, C, H, W]
                    noisy_center = seqn_val.squeeze_()[center_idx:center_idx+1].cpu()  # [1, C, H, W]
                    denoised_center = out_val[center_idx:center_idx+1].cpu()  # [1, C, H, W]
                    
                    # Make grids for visualization
                    clean_vis = torchvision.utils.make_grid(clean_center.cpu().clamp(0, 1), normalize=False)
                    noisy_vis = torchvision.utils.make_grid(noisy_center.clamp(0, 1), normalize=False)
                    denoised_vis = torchvision.utils.make_grid(denoised_center.clamp(0, 1), normalize=False)
                    
                    # Log validation images
                    self.writer.add_image('Validation/Clean', clean_vis, self.epoch)
                    self.writer.add_image('Validation/Noisy', noisy_vis, self.epoch)
                    self.writer.add_image('Validation/Student_Denoised', denoised_vis, self.epoch)
                    
                    # Create comparison
                    comparison = torch.cat([noisy_vis, denoised_vis, clean_vis], dim=2)
                    self.writer.add_image('Validation/Comparison_Noisy_Denoised_Clean', comparison, self.epoch)
                    
                    validation_logged = True
    
        
        psnr_val /= len(self.val_loader)
        t2 = time.time()
        
        # Log validation metrics
        if self.writer is not None:
            self.writer.add_scalar('Validation/PSNR', psnr_val, self.epoch)
            self.writer.add_scalar('Validation/Best_PSNR', self.best_psnr, self.epoch)
        
        # Check if current model is the best
        if psnr_val > self.best_psnr:
            self.best_psnr = psnr_val
            self.is_best = True
        else:
            self.is_best = False
        
        status_msg = f"\n[epoch {self.epoch}] psnr_val: {psnr_val:.4f} (best: {self.best_psnr:.4f}), on {t2-t1:.2f} sec"
        tqdm.write(status_msg)

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="On-the-fly Distillation Training with ShiftNet")
    parser.add_argument("--config", type=str, default="./configs/distill.yaml", help="path to config file")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Convert to namespace
    for k, v in config.items():
        setattr(args, k, v)
    
    # Normalize noise values
    if hasattr(args, 'val_noiseL'):
        args.val_noiseL /= 255.
    if hasattr(args, 'noise_ival'):
        args.noise_ival[0] /= 255.
        args.noise_ival[1] /= 255.
    
    # Create trainer and start training
    trainer = OnFlyDistillationTrainer(args)
    trainer.train()
    print("Training complete!")

if __name__ == "__main__":
    print("Script started")
    main()


