'''Implements a sequence dataloader for loading image sequences from folders.'''
import os
import cv2
import glob
import random

import numpy as np
from PIL import Image
from random import choices # requires Python >= 3.6

import torch
import torch.utils.data as data
import torchvision.transforms.functional as f
from torch.utils.data.dataset import Dataset

from .utils import *


class DVDDataset(data.Dataset):
    '''Loads sequences of images from folders.

    Args:
        root_dir (str): Path to the main directory containing sequence folders.
        sequence_length (int): Number of frames per sequence.
        crop_size (int or tuple): Desired output size (height, width). If int, square crop.
        transform (callable, optional): Optional transform to be applied on a sample.
    '''
    def __init__(self, root_dir, sequence_length=5, crop_size=96, step = 3, transform=None):
        super().__init__()

        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size)
        else:
            self.crop_size = crop_size # Should be (h, w)
        if self.crop_size[0] <= 0 or self.crop_size[1] <= 0:
             raise ValueError("crop_size must be positive")

        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.sequences = []
        self.step = step
        self.transform = transform

        sequence_folders = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

        for seq_folder in sequence_folders:
            seq_path_val = os.path.join(root_dir, seq_folder)
            # Pre-calculate image paths for the current sequence folder
            img_paths_for_folder = sorted(glob.glob(os.path.join(seq_path_val, '*.png')))
            
            num_frames = len(img_paths_for_folder)

            if num_frames >= self.sequence_length:
                for i in range(0, num_frames - self.sequence_length + 1, self.step):
                    # Store the list of image paths, start index, and original sequence path
                    self.sequences.append((img_paths_for_folder, i, seq_path_val))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        '''Retrieves a sequence item.

        Returns:
            torch.Tensor: Sequence tensor of shape [F, C, H, W], normalized to [0, 1].
        '''
        img_paths, start_index, seq_path_for_error_msg = self.sequences[index]

        sequence_imgs = []
        # Load the first image to get dimensions for random crop
        first_img_path = img_paths[start_index]
        first_img = Image.open(first_img_path).convert('RGB')
        width, height = first_img.size

        # Get random crop parameters (consistent across the sequence)
        crop_h, crop_w = self.crop_size
        if crop_w > width or crop_h > height:
             raise ValueError(f"Crop size ({crop_h}, {crop_w}) larger than image size ({height}, {width}) for sequence {seq_path_for_error_msg}")
        top = random.randint(0, height - crop_h)
        left = random.randint(0, width - crop_w)

        for i in range(self.sequence_length):
            img_path = img_paths[start_index + i]
            img = Image.open(img_path).convert('RGB')

            # Apply the same crop to all images in the sequence
            img_cropped = f.crop(img, top, left, crop_h, crop_w)

            # Convert to tensor and normalize to [0, 1]
            img_tensor = f.to_tensor(img_cropped).float()
            sequence_imgs.append(img_tensor)

        sequence_tensor = torch.stack(sequence_imgs, dim=0)  # Shape: [F, C, H, W]

        if self.transform:
            sequence_tensor = self.transform(sequence_tensor)
        
        return sequence_tensor



class ValDataset(Dataset):
    """Validation dataset. Loads all the images in the dataset folder on memory.
    """
    def __init__(self, valsetdir=None, gray_mode=False, num_input_frames=15):
        self.gray_mode = gray_mode

        VALSEQPATT = '*' # pattern for name of validation sequence      
        # Look for subdirs with individual sequences
        seqs_dirs = sorted(glob.glob(os.path.join(valsetdir, VALSEQPATT)))      
        # open individual sequences and append them to the sequence list
        sequences = []
        for seq_dir in seqs_dirs:
            seq, _, _ = open_sequence(seq_dir, gray_mode, expand_if_needed=False, \
            				 max_num_fr=num_input_frames)
            # seq is [num_frames, C, H, W]
            sequences.append(seq)       
        self.sequences = sequences

    def __getitem__(self, index):
        return torch.from_numpy(self.sequences[index])

    def __len__(self):
        return len(self.sequences)
