import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode  # For Resize
from pathlib import Path
import glob
import natsort  # For natural sorting of filenames
from random import choices  # For weighted random selection of augmentations

from .utils import *


class DVDDataset(Dataset):
    def __init__(
        self,
        root_dir,
        sequence_length=5,
        ctrl_fr_idx=2,
        channels=3,
        apply_sequence_augmentations=True,
        crop_size=(96, 96),
    ):
        """
        Args:
            root_dir (string): Directory with all the images or parent of scene directories.
            sequence_length (int): Number of frames per sequence. Default is 5.
            ctrl_fr_idx (int): Index of the center frame in the sequence to be used as ground truth.
                               Default is 2 (the 3rd frame in a 0-indexed sequence of 5).
            channels (int): Number of image channels (e.g., 3 for RGB, 1 for grayscale).
            apply_sequence_augmentations (bool): Whether to apply the sequence-level geometric/noise augmentations.
            crop_size (tuple): Tuple (height, width) specifying the size of the crop. Default is None (no cropping).
        """

        self.root_dir = Path(root_dir)
        self.sequence_length = sequence_length
        self.ctrl_fr_idx = ctrl_fr_idx
        self.channels = channels
        self.apply_sequence_augmentations = apply_sequence_augmentations
        self.crop_size = crop_size
        self.image_paths_sequences = []

        self.transform = T.Compose([T.ToTensor(),])

        all_scene_dirs = [d for d in self.root_dir.iterdir() if d.is_dir()]
        if not all_scene_dirs:
            all_scene_dirs = [self.root_dir]

        for scene_dir in all_scene_dirs:
            scene_image_paths = natsort.natsorted(
                [p for p in scene_dir.glob("*.png") if p.is_file()]
            )
            if not scene_image_paths:
                continue
            for i in range(len(scene_image_paths) - self.sequence_length + 1):
                sequence = scene_image_paths[i : i + self.sequence_length]
                self.image_paths_sequences.append(sequence)

        if not self.image_paths_sequences:
            raise RuntimeError(f"No image sequences found in {self.root_dir}")
        print(f"Found {len(self.image_paths_sequences)} sequences.")

        # Define sequence-level augmentation operations
        self._do_nothing = lambda x: x
        self._flipud = lambda x: torch.flip(x, dims=[-2])  # H dimension
        self._rot90 = lambda x: torch.rot90(x, k=1, dims=[-2, -1])  # H, W dimensions
        self._rot90_flipud = lambda x: torch.flip(
            torch.rot90(x, k=1, dims=[-2, -1]), dims=[-2]
        )
        self._rot180 = lambda x: torch.rot90(x, k=2, dims=[-2, -1])
        self._rot180_flipud = lambda x: torch.flip(
            torch.rot90(x, k=2, dims=[-2, -1]), dims=[-2]
        )
        self._rot270 = lambda x: torch.rot90(x, k=3, dims=[-2, -1])
        self._rot270_flipud = lambda x: torch.flip(
            torch.rot90(x, k=3, dims=[-2, -1]), dims=[-2]
        )
        self._add_csnt = lambda x: (
            x + torch.normal(mean=torch.zeros_like(x), std=(5.0 / 255.0))
        ).clamp(0.0, 1.0)

        self.aug_list = [
            self._do_nothing,
            self._flipud,
            self._rot90,
            self._rot90_flipud,
            self._rot180,
            self._rot180_flipud,
            self._rot270,
            self._rot270_flipud,
            self._add_csnt,
        ]
        self.w_aug = [
            32,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
        ]

    def __len__(self):
        return len(self.image_paths_sequences)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sequence_paths = self.image_paths_sequences[idx]
        frames = []
        pil_mode = "RGB" if self.channels == 3 else "L" if self.channels == 1 else None

        for img_path in sequence_paths:
            img = Image.open(img_path).convert(pil_mode)
            # convert to tensor
            img = self.transform(img)
            frames.append(img)

        try:
            if not all(isinstance(f, torch.Tensor) for f in frames):
                raise TypeError(
                    "All frames must be PyTorch Tensors before stacking. Ensure ToTensor() is in your transform."
                )
            sequence_tensor = torch.stack(frames)
        except Exception as e:
            print(
                f"Error stacking frames for sequence {idx} (paths: {sequence_paths}): {e}"
            )
            return None

        if self.crop_size:
            crop_h, crop_w = self.crop_size
            _, _, h, w = sequence_tensor.shape
            if h < crop_h or w < crop_w:
                raise ValueError(
                    f"Crop size ({crop_h}, {crop_w}) is larger than image dimensions ({h}, {w})."
                )
            top = torch.randint(0, h - crop_h + 1, (1,)).item()
            left = torch.randint(0, w - crop_w + 1, (1,)).item()
            sequence_tensor = sequence_tensor[
                :, :, top : top + crop_h, left : left + crop_w
            ]

        processed_sequence = sequence_tensor.view(
            -1, sequence_tensor.size(-2), sequence_tensor.size(-1)
        )

        if self.apply_sequence_augmentations:
            chosen_transform_func = choices(self.aug_list, self.w_aug)[0]
            processed_sequence = chosen_transform_func(processed_sequence)

        start_channel_idx = self.channels * self.ctrl_fr_idx
        end_channel_idx = start_channel_idx + self.channels

        gt_frame = processed_sequence[start_channel_idx:end_channel_idx, :, :]
        return processed_sequence, gt_frame


class ValDataset(Dataset):
    """Validation dataset. Loads all the images in the dataset folder on memory."""

    def __init__(self, valsetdir=None, gray_mode=False, num_input_frames=15):
        self.gray_mode = gray_mode

        VALSEQPATT = "*"  # pattern for name of validation sequence
        # Look for subdirs with individual sequences
        seqs_dirs = sorted(glob.glob(os.path.join(valsetdir, VALSEQPATT)))
        # open individual sequences and append them to the sequence list
        sequences = []
        for seq_dir in seqs_dirs:
            seq, _, _ = open_sequence(
                seq_dir, gray_mode, expand_if_needed=False, max_num_fr=num_input_frames
            )
            # seq is [num_frames, C, H, W]
            sequences.append(seq)
        self.sequences = sequences

    def __getitem__(self, index):
        return torch.from_numpy(self.sequences[index])

    def __len__(self):
        return len(self.sequences)
