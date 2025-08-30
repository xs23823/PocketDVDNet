import yaml
import torch
import argparse

from trainer import Trainer
from utils.prefetcher import PrefetchDataLoader, CPUPrefetcher

# Replace specific model import with general student import
from student import PocketDVDnet, PocketDVDnet7
from dataloaders.fastdvdnet import ValDataset
from dataloaders.distill_dataloader import DistillationDataset


def main(args):
    torch.manual_seed(args.seed)

    if torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator()
        print(f"Using {device} accelerator")
    else:
        print("No accelerator found, using CPU")
        device = torch.device("cpu")

    # Get sequence length from config
    sequence_length = getattr(args, 'sequence_length', 5)  # Default to 5 if not specified
    print(f"Using sequence length: {sequence_length}")

    # validation dataset (unchanged)
    val_dataset = ValDataset(valsetdir=args.valset_dir, gray_mode=False, num_input_frames=sequence_length)
    
    # training dataset
    if getattr(args, 'use_distillation', False):
        print("loading distillation dataset with precomputed outputs")
        train_dataset = DistillationDataset(
            noisy_dir=args.noisy_dir,
            teacher_dir=args.teacher_dir,
            gt_dir=args.gt_dir,
            sequence_length=sequence_length,
            ctrl_fr_idx=sequence_length // 2,  # Center frame
            crop_size=args.patch_size
        )
    else:
        # original training dataset setup
        from dataloaders.fastdvdnet import DVDDataset, Sampler
        dvd = DVDDataset(
            root_dir=args.trainset_dir,
            sequence_length=sequence_length,
            ctrl_fr_idx=sequence_length // 2,  # Center frame
            crop_size=args.patch_size,
        )
        # additional datasets if needed
        train_dataset = dvd

    # loaders
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=1,
        num_workers=2,
        pin_memory=True,
        shuffle=False,
    )
    
    train_loader = PrefetchDataLoader(
        args.num_prefetch_queue,
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
        shuffle=True,
        drop_last=True,
    )
    
    prefetcher = CPUPrefetcher(train_loader)

    # model - dynamically select based on sequence_length
    if sequence_length == 7:
        print("Using PocketDVDnet7 model (7-frame)")
        model = PocketDVDnet7(num_input_frames=sequence_length).to(device)
    elif sequence_length == 5:
        print("Using PocketDVDnet model (5-frame)")
        model = PocketDVDnet(num_input_frames=sequence_length).to(device)
    else:
        raise ValueError(f"Unsupported sequence length: {sequence_length}. Must be 5 or 7.")
    
    # optionally load pretrained weights for fine-tuning
    if getattr(args, 'pretrained_path', None):
        print(f"loading pretrained model from {args.pretrained_path}")
        ckpt = torch.load(args.pretrained_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
    
    # trainer
    trainer = Trainer(args, model, prefetcher, val_loader)
    
    # compile or not
    if getattr(args, 'use_obproxsg', False):
        trainer.train()
    else:
        torch.compile(trainer.train(), mode="default")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PocketDVDnet Training")
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/adam.yaml",
        help="path to YAML config file",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    for key, val in cfg.items():
        setattr(args, key, val)

    # normalize noise values
    if hasattr(args, 'val_noiseL'):
        args.val_noiseL /= 255.
    if hasattr(args, 'noise_ival'):
        args.noise_ival[0] /= 255.
        args.noise_ival[1] /= 255.

    main(args)
