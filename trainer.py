import os
import glob
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

from torch.optim.lr_scheduler import StepLR
from utils.lr_scheduler import MultiStepRestartLR, CosineAnnealingRestartLR


class Trainer:
    def __init__(self, args, model, prefetcher, val_loader, start_epoch=0):

        self.args = args
        self.model = model
        self.prefetcher = prefetcher
        self.val_loader = val_loader

        self.device = next(model.parameters()).device
        self.epoch = start_epoch
        self.iteration = 0
        self.best_acc = 0
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

        self.optimizer = torch.optim.Adadelta(
            optim_params,
        )

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
            self.scheduler = StepLR(self.optimizer, step_size=1, gamma=self.args.gamma)


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
            ckpt = torch.load(os.path.join(self.args.out_dir, "latest.pt"), map_location=self.device)
            latest_epoch = ckpt.get("epoch", None)
        else:
            latest_epoch = None

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

    def save(self):
        """Save latest checkpoint and, if flagged, update best checkpoint."""
        # bundle everything into one dict
        ckpt = {
            "epoch": self.epoch,
            "iteration": self.iteration,
            "model_state": self.model.state_dict(),
            "optim_state": self.optimizer.state_dict(),
            "sched_state": self.scheduler.state_dict(),
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
            self.update_learning_rate()
            if self.epoch % self.args.save_freq == 0:
                self.save()

        pbar.close()
        tqdm.write("\nTraining complete.")

    def train_epoch(self, pbar):
        """
        Process input and calculate loss every training epoch
        """

        self.model.train()
        tqdm.write(f"Training epoch {self.epoch}...")

        while True:
            batch = self.prefetcher.next()
            if batch is None:
                break

            data, target = batch
            self.iteration += 1

            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                output = self.model(data)
                loss = F.nll_loss(output, target)

            # Backpropagation
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.add_summary(self.writer, "loss", loss.item())

            # Console logs
            pbar.update(1)
            if self.iteration % 10 == 0:
                self.current_lr = self.get_lr()
                pbar.set_description((f"LR: {self.current_lr} Loss: {loss.item():.3f}"))


        # Clean up
        torch.cuda.empty_cache()

        tqdm.write(
            "Train Epoch: {} [{}/{} ({:.0f}%)] - [Loss: {:.6f}]".format(
                self.epoch,
                self.iteration,
                self.args.iterations,
                100.0, #* batch_idx / len(train_loader),
                loss.item(),
            )
        )

    def validate(self):
        """Validate the model."""
        self.model.eval()

        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.nll_loss(
                    output, target, reduction="sum"
                ).item()  # sum up batch loss
                pred = output.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        self.best_acc = max(correct / len(self.val_loader.dataset), self.best_acc)
        self.is_best = correct >= self.best_acc

        test_loss /= len(self.val_loader.dataset)

        tqdm.write(
            "Validation Results: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss,
                correct,
                len(self.val_loader.dataset),
                100.0 * correct / len(self.val_loader.dataset),
            )
        )
