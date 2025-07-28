import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from pathlib import Path
import natsort


class DistillationDataset(Dataset):
    """dataset for distillation with precomputed teacher outputs"""
    
    def __init__(self, 
                 noisy_dir,
                 teacher_dir, 
                 gt_dir,
                 sequence_length=7,
                 ctrl_fr_idx=3,
                 crop_size=None):
        """
        args:
            noisy_dir: path to noisy input sequences (what pacnet saw)
            teacher_dir: path to teacher (pacnet) denoised outputs
            gt_dir: path to ground truth clean sequences
            sequence_length: number of frames per sequence
            ctrl_fr_idx: index of center frame
            crop_size: optional crop size for patches
        """
        self.noisy_dir = Path(noisy_dir)
        self.teacher_dir = Path(teacher_dir)
        self.gt_dir = Path(gt_dir)
        self.sequence_length = sequence_length
        self.ctrl_fr_idx = ctrl_fr_idx
        self.crop_size = (crop_size, crop_size) if isinstance(crop_size, int) else crop_size
        
        self.transform = T.ToTensor()
        self.sequences = []
        
        # find all sequences
        for seq_name in os.listdir(self.noisy_dir):
            noisy_seq_dir = self.noisy_dir / seq_name
            teacher_seq_dir = self.teacher_dir / seq_name
            gt_seq_dir = self.gt_dir / seq_name
            
            if not all(p.is_dir() for p in [noisy_seq_dir, teacher_seq_dir, gt_seq_dir]):
                continue
                
            # get frame paths for this sequence
            noisy_frames = natsort.natsorted(list(noisy_seq_dir.glob('*.png')))
            teacher_frames = natsort.natsorted(list(teacher_seq_dir.glob('*.png')))
            gt_frames = natsort.natsorted(list(gt_seq_dir.glob('*.png')))
            
            # check all have same number of frames
            if not (len(noisy_frames) == len(teacher_frames) == len(gt_frames)):
                print(f"warning: sequence {seq_name} has mismatched frame counts")
                continue
            
            # create sequences with sliding window
            for i in range(len(noisy_frames) - self.sequence_length + 1):
                self.sequences.append({
                    'noisy': noisy_frames[i:i+self.sequence_length],
                    'teacher': teacher_frames[i+self.ctrl_fr_idx],  # only need center frame
                    'gt': gt_frames[i+self.ctrl_fr_idx]  # only need center frame
                })
                
        print(f"loaded {len(self.sequences)} sequences for distillation")
        
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
        
        # reshape noisy sequence for model input
        noisy_input = noisy_seq.view(-1, *noisy_seq.shape[-2:])  # [F*C, H, W]
        
        return {
            'noisy_input': noisy_input,
            'teacher_target': teacher_frame,
            'gt_target': gt_frame
        }