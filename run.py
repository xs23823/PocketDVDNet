import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import yaml


from models.net import Net
from trainer import Trainer
from utils.prefetcher import PrefetchDataLoader, CPUPrefetcher




# def train(args, model, device, train_loader, optimizer, epoch):
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = F.nll_loss(output, target)
#         loss.backward()
#         optimizer.step()
#         if batch_idx % args.log_interval == 0:
   


def main(args):

    torch.manual_seed(args.seed)

    if torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator()
        print(f"Using {device} accelerator")
    else:
        print("No accelerator found, using CPU")
        device = torch.device("cpu")


    # Datasets
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    train_dataset = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    val_dataset = datasets.MNIST('../data', train=False,
                       transform=transform)
    
    # Loaders
    val_loader  = torch.utils.data.DataLoader(
        **dict(
            dataset=val_dataset,
            batch_size=args.test_batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=False,
        )
    )
    train_loader = PrefetchDataLoader(
        args.num_prefetch_queue, 
        **dict(
            dataset=train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=True,
            drop_last=True
        )
    )
    prefetcher = CPUPrefetcher(train_loader)


    model = Net().to(device)

    trainer = Trainer(args, model, prefetcher, val_loader)
    torch.compile(trainer.train(), mode="default")


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--config', type=str, default="./configs/net.yaml",
                        help='path to YAML config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    for key, val in cfg.items():
        setattr(args, key, val)
    main(args)