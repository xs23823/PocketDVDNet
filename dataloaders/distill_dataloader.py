import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from pathlib import Path
import natsort
import random
import numpy as np


class DistillationDataset(Dataset):
    """dataset for distillation with precomputed teacher outputs"""
    
    def __init__(self, 
                 noisy_dir,
                 teacher_dir, 
                 gt_dir,
                 sequence_length=7,
                 ctrl_fr_idx=3,
                 crop_size=None,
                 apply_augmentations=True):
        """
        args:
            noisy_dir: path to noisy input sequences (what pacnet saw)
            teacher_dir: path to teacher (pacnet) denoised outputs
            gt_dir: path to ground truth clean sequences
            sequence_length: number of frames per sequence
            ctrl_fr_idx: index of center frame
            crop_size: optional crop size for patches
            apply_augmentations: whether to apply data augmentations
        """
        self.noisy_dir = Path(noisy_dir)
        self.teacher_dir = Path(teacher_dir)
        self.gt_dir = Path(gt_dir)
        self.sequence_length = sequence_length
        self.ctrl_fr_idx = ctrl_fr_idx
        self.crop_size = (crop_size, crop_size) if isinstance(crop_size, int) else crop_size
        self.apply_augmentations = apply_augmentations
        
        self.transform = T.ToTensor()
        self.sequences = []
        
        # verify directories exist
        for dir_path, name in [(self.noisy_dir, 'noisy'), 
                                (self.teacher_dir, 'teacher'), 
                                (self.gt_dir, 'gt')]:
            if not dir_path.exists():
                raise ValueError(f"{name} directory does not exist: {dir_path}")
            print(f"found {name} directory: {dir_path}")
        
        # find all sequences
        seq_names = [d.name for d in self.noisy_dir.iterdir() if d.is_dir()]
        print(f"found {len(seq_names)} sequence directories in noisy_dir")
        
        if len(seq_names) == 0:
            print("warning: no subdirectories found. checking for images directly in directories...")
            # try loading images directly from root directories
            noisy_frames = natsort.natsorted(list(self.noisy_dir.glob('*.png')) + 
                                            list(self.noisy_dir.glob('*.jpg')))
            teacher_frames = natsort.natsorted(list(self.teacher_dir.glob('*.png')) + 
                                             list(self.teacher_dir.glob('*.jpg')))
            gt_frames = natsort.natsorted(list(self.gt_dir.glob('*.png')) + 
                                        list(self.gt_dir.glob('*.jpg')))
            
            print(f"found {len(noisy_frames)} noisy, {len(teacher_frames)} teacher, {len(gt_frames)} gt frames")
            
            if len(noisy_frames) > 0:
                self._process_sequence("root", noisy_frames, teacher_frames, gt_frames)
        else:
            # process each sequence directory
            for seq_name in seq_names:
                noisy_seq_dir = self.noisy_dir / seq_name
                teacher_seq_dir = self.teacher_dir / seq_name
                gt_seq_dir = self.gt_dir / seq_name
                
                if not all(p.is_dir() for p in [noisy_seq_dir, teacher_seq_dir, gt_seq_dir]):
                    print(f"warning: sequence {seq_name} missing in some directories")
                    continue
                
                # get frame paths for this sequence
                noisy_frames = natsort.natsorted(list(noisy_seq_dir.glob('*.png')) + 
                                                list(noisy_seq_dir.glob('*.jpg')))
                teacher_frames = natsort.natsorted(list(teacher_seq_dir.glob('*.png')) + 
                                                 list(teacher_seq_dir.glob('*.jpg')))
                gt_frames = natsort.natsorted(list(gt_seq_dir.glob('*.png')) + 
                                            list(gt_seq_dir.glob('*.jpg')))
                
                self._process_sequence(seq_name, noisy_frames, teacher_frames, gt_frames)
        
        if len(self.sequences) == 0:
            raise ValueError("no valid sequences found! check your directory structure and file formats")
            
        print(f"loaded {len(self.sequences)} sequences for distillation")
        
        # setup augmentation functions (matching original DVDDataset)
        self.aug_transforms = [
            lambda x: x,  # do nothing
            lambda x: torch.flip(x, dims=[-2]),  # vertical flip
            lambda x: torch.rot90(x, k=1, dims=[-2, -1]),  # rotate 90
            lambda x: torch.flip(torch.rot90(x, k=1, dims=[-2, -1]), dims=[-2]),  # rotate 90 + flip
            lambda x: torch.rot90(x, k=2, dims=[-2, -1]),  # rotate 180
            lambda x: torch.flip(torch.rot90(x, k=2, dims=[-2, -1]), dims=[-2]),  # rotate 180 + flip
            lambda x: torch.rot90(x, k=3, dims=[-2, -1]),  # rotate 270
            lambda x: torch.flip(torch.rot90(x, k=3, dims=[-2, -1]), dims=[-2]),  # rotate 270 + flip
        ]
        # weights matching original (32, 12, 12, 12, 12, 12, 12, 12)
        self.aug_weights = [32, 12, 12, 12, 12, 12, 12, 12]
    
    def _process_sequence(self, seq_name, noisy_frames, teacher_frames, gt_frames):
        """process a single sequence"""
        # check all have same number of frames
        if not (len(noisy_frames) == len(teacher_frames) == len(gt_frames)):
            print(f"warning: sequence {seq_name} has mismatched frame counts: "
                  f"noisy={len(noisy_frames)}, teacher={len(teacher_frames)}, gt={len(gt_frames)}")
            return
        
        if len(noisy_frames) < self.sequence_length:
            print(f"warning: sequence {seq_name} has only {len(noisy_frames)} frames, "
                  f"need at least {self.sequence_length}")
            return
        
        # create sequences with sliding window
        for i in range(len(noisy_frames) - self.sequence_length + 1):
            self.sequences.append({
                'noisy': noisy_frames[i:i+self.sequence_length],
                'teacher': teacher_frames[i+self.ctrl_fr_idx],  # only need center frame
                'gt': gt_frames[i+self.ctrl_fr_idx]  # only need center frame
            })
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq_data = self.sequences[idx]
        
        # load noisy sequence
        noisy_frames = []
        for frame_path in seq_data['noisy']:
            img = Image.open(frame_path).convert('RGB')
            img = self.transform(img)
            noisy_frames.append(img)
        noisy_seq = torch.stack(noisy_frames)  # [F, C, H, W]
        
        # load teacher output (center frame only)
        teacher_img = Image.open(seq_data['teacher']).convert('RGB')
        teacher_frame = self.transform(teacher_img)  # [C, H, W]
        
        # load gt (center frame only)
        gt_img = Image.open(seq_data['gt']).convert('RGB')
        gt_frame = self.transform(gt_img)  # [C, H, W]
        
        # apply consistent crops if specified
        if self.crop_size:
            h, w = noisy_seq.shape[-2:]
            crop_h, crop_w = self.crop_size
            
            if h >= crop_h and w >= crop_w:
                # random crop position
                top = torch.randint(0, h - crop_h + 1, (1,)).item()
                left = torch.randint(0, w - crop_w + 1, (1,)).item()
                
                # apply same crop to all
                noisy_seq = noisy_seq[:, :, top:top+crop_h, left:left+crop_w]
                teacher_frame = teacher_frame[:, top:top+crop_h, left:left+crop_w]
                gt_frame = gt_frame[:, top:top+crop_h, left:left+crop_w]
        
        # apply augmentations if enabled
        if self.apply_augmentations:
            # choose augmentation
            aug_idx = random.choices(range(len(self.aug_transforms)), 
                                   weights=self.aug_weights, k=1)[0]
            aug_fn = self.aug_transforms[aug_idx]
            
            # apply same augmentation to all
            noisy_seq = torch.stack([aug_fn(frame) for frame in noisy_seq])
            teacher_frame = aug_fn(teacher_frame)
            gt_frame = aug_fn(gt_frame)
        
        # reshape noisy sequence for model input
        noisy_input = noisy_seq.view(-1, *noisy_seq.shape[-2:])  # [F*C, H, W]
        
        return {
            'noisy_input': noisy_input,
            'teacher_target': teacher_frame,
            'gt_target': gt_frame
        }