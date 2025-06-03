import yaml
import torch
import argparse

from trainer import Trainer
from utils.prefetcher import PrefetchDataLoader, CPUPrefetcher

from models.fastdvdnet import FastDVDnet
from dataloaders.fastdvdnet import DVDDataset, ValDataset, Sampler

def main(args):

    torch.manual_seed(args.seed)

    if torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator()
        print(f"Using {device} accelerator")
    else:
        print("No accelerator found, using CPU")
        device = torch.device("cpu")

    # Datasets
    val_dataset = ValDataset(valsetdir=args.valset_dir, gray_mode=False)
    dvd = DVDDataset(
        root_dir=args.trainset_dir,
        crop_size=args.patch_size,
    )
    bvidvc = DVDDataset(
        root_dir="/home/imogend/Documents/Data/PNG_TRAINING_SEQUENCES",
        crop_size=args.patch_size,
    )
    train_sampler = Sampler([bvidvc, dvd], iter = True)

    # Loaders
    val_loader = torch.utils.data.DataLoader(
        **dict(
            dataset=val_dataset,
            batch_size=1,
            num_workers=2,
            pin_memory=True,
            shuffle=False,
        )
    )
    train_loader = PrefetchDataLoader(
        args.num_prefetch_queue,
        **dict(
            dataset=train_sampler,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=False,
            shuffle=True,
            drop_last=True,
        ),
    )
    prefetcher = CPUPrefetcher(train_loader)

    model = FastDVDnet().to(device)

    trainer = Trainer(args, model, prefetcher, val_loader)
    
    if getattr(args, 'use_obproxsg', False):
        trainer.train()  # No compile for OBProxSG
    else:
        torch.compile(trainer.train(), mode="default")


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/fastdvdnet.yaml",
        help="path to YAML config file",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    for key, val in cfg.items():
        setattr(args, key, val)

    args.val_noiseL /= 255.
    args.noise_ival[0] /= 255.
    args.noise_ival[1] /= 255.

    main(args)
