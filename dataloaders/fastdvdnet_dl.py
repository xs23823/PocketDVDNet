'''Implements a sequence dataloader for loading image sequences from folders.'''
import os
import random
from glob import glob
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms.functional as f


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

        sequence_folders = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

        for seq_folder in sequence_folders:
            seq_path = os.path.join(root_dir, seq_folder)
            img_paths = sorted(glob(os.path.join(seq_path, '*.png')))
            
            num_frames = len(img_paths)

            if num_frames >= self.sequence_length:
                for i in range(0, num_frames - self.sequence_length + 1, self.step):
                    self.sequences.append((seq_path, i))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        '''Retrieves a sequence item.

        Args:
            index (int): Index of the sequence.

        Returns:
            torch.Tensor: Sequence tensor of shape [F, C, H, W], normalized to [0, 1].
        '''
        seq_path, start_index = self.sequences[index]
        img_paths = img_paths = sorted(glob(os.path.join(seq_path, '*.png')))

        sequence_imgs = []
        # Load the first image to get dimensions for random crop
        first_img = Image.open(img_paths[start_index]).convert('RGB')
        width, height = first_img.size

        # Get random crop parameters (consistent across the sequence)
        crop_h, crop_w = self.crop_size
        if crop_w > width or crop_h > height:
             raise ValueError(f"Crop size ({crop_h}, {crop_w}) larger than image size ({height}, {width}) for sequence {seq_path}")
        top = random.randint(0, height - crop_h)
        left = random.randint(0, width - crop_w)

        for i in range(self.sequence_length):
            img_path = img_paths[start_index + i]
            try:
                img = Image.open(img_path).convert('RGB')
                # Apply the same crop to all images in the sequence
                img_cropped = f.crop(img, top, left, crop_h, crop_w)
                # Convert to tensor and normalize to [0, 1]
                img_tensor = f.to_tensor(img_cropped).float()

                sequence_imgs.append(img_tensor)
            except Exception as e:
                print(f"Error loading or processing image {img_path}: {e}")
                return None

        # Stack images into a sequence tensor [F, C, H, W]
        sequence_tensor = torch.stack(sequence_imgs, dim=0)

        return sequence_tensor

class Sampler(torch.utils.data.Dataset):
    def __init__(self, datasets, p_datasets=None, iter=False, samples_per_epoch=1000):
        self.datasets = datasets
        self.len_datasets = np.array([len(dataset) for dataset in self.datasets])
        self.p_datasets = p_datasets
        self.iter = iter

        if p_datasets is None:
            self.p_datasets = self.len_datasets / np.sum(self.len_datasets)

        self.samples_per_epoch = samples_per_epoch

        self.accum = [
            0,
        ]
        for i, length in enumerate(self.len_datasets):
            self.accum.append(self.accum[-1] + self.len_datasets[i])

    def __getitem__(self, index):
        if self.iter:
            # iterate through all datasets
            for i in range(len(self.accum)):
                if index < self.accum[i]:
                    return self.datasets[i - 1].__getitem__(index - self.accum[i - 1])
        else:
            # first sample a dataset
            dataset = random.choices(self.datasets, self.p_datasets)[0]
            # sample a sequence from the dataset
            return dataset.__getitem__(random.randint(0, len(dataset) - 1))

    def __len__(self):
        if self.iter:
            return int(np.sum(self.len_datasets))
        else:
            return self.samples_per_epoch