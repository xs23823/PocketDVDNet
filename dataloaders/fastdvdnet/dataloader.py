import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

from pathlib import Path
import glob
import natsort  # For natural sorting of filenames
import random
import numpy as np
import albumentations as A
import cv2 # Added for albumentations Rotate interpolation

from .utils import *


class DVDDataset(Dataset):
    def __init__(
        self,
        root_dir,
        sequence_length=5, # Default to 5, will be overridden by config
        ctrl_fr_idx=None,  # If None, will calculate based on sequence_length
        channels=3,
        apply_sequence_augmentations=True,
        crop_size=96,
    ):
        """
        Args:
            root_dir (string): Directory with all the images or parent of scene directories.
            sequence_length (int): Number of frames per sequence. Default is 5.
            ctrl_fr_idx (int): Index of the center frame in the sequence to be used as ground truth.
                              If None, will be calculated as sequence_length // 2.
            channels (int): Number of image channels (e.g., 3 for RGB, 1 for grayscale).
            apply_sequence_augmentations (bool): Whether to apply the sequence-level geometric/noise augmentations.
            crop_size (tuple): Tuple (height, width) specifying the size of the crop. Default is None (no cropping).
        """

        self.root_dir = Path(root_dir)
        self.sequence_length = sequence_length
        
        # Calculate center frame index if not provided
        if ctrl_fr_idx is None:
            self.ctrl_fr_idx = self.sequence_length // 2
        else:
            self.ctrl_fr_idx = ctrl_fr_idx
            
        self.channels = channels
        self.apply_sequence_augmentations = apply_sequence_augmentations
        self.crop_size = (crop_size, crop_size) if isinstance(crop_size, int) else crop_size
        self.image_paths_sequences = []

        self.transform = T.Compose([T.ToTensor(),])

        all_scene_dirs = [d for d in self.root_dir.iterdir() if d.is_dir()]
        if not all_scene_dirs:
            all_scene_dirs = [self.root_dir]

        for scene_dir in all_scene_dirs:
            scene_image_paths = natsort.natsorted(
                [p for ext in ("*.png", "*.jpg", "*.jpeg") for p in scene_dir.glob(ext) if p.is_file()]
            )
            if not scene_image_paths:
                continue
            for i in range(len(scene_image_paths) - self.sequence_length + 1):
                sequence = scene_image_paths[i : i + self.sequence_length]
                self.image_paths_sequences.append(sequence)

        if not self.image_paths_sequences:
            raise RuntimeError(f"No image sequences found in {self.root_dir}")

        # Albumentations equivalents for the original augmentations
        # Interpolation for Rotate: cv2.INTER_LINEAR is default and good.
        # torch.rot90 is exact; cv2.INTER_NEAREST is closer for integer grids but blocky.
        # Using cv2.INTER_LINEAR for smoother results.
        self.A_do_nothing = A.Compose([])
        self.A_flipud = A.Compose([A.VerticalFlip(p=1.0)])
        self.A_rot90 = A.Compose([A.Rotate(limit=(90, 90), interpolation=cv2.INTER_LINEAR, p=1.0)])
        self.A_rot90_flipud = A.Compose([
            A.Rotate(limit=(90, 90), interpolation=cv2.INTER_LINEAR, p=1.0),
            A.VerticalFlip(p=1.0)
        ])
        self.A_rot180 = A.Compose([A.Rotate(limit=(180, 180), interpolation=cv2.INTER_LINEAR, p=1.0)])
        self.A_rot180_flipud = A.Compose([
            A.Rotate(limit=(180, 180), interpolation=cv2.INTER_LINEAR, p=1.0),
            A.VerticalFlip(p=1.0)
        ])
        self.A_rot270 = A.Compose([A.Rotate(limit=(270, 270), interpolation=cv2.INTER_LINEAR, p=1.0)])
        self.A_rot270_flipud = A.Compose([
            A.Rotate(limit=(270, 270), interpolation=cv2.INTER_LINEAR, p=1.0),
            A.VerticalFlip(p=1.0)
        ])
        
        # Original std was 5.0/255.0 for float images in [0,1] range.
        # Albumentations GaussNoise var_limit is (variance_min, variance_max).
        # Variance = std^2.
        noise_std = 5.0 / 255.0
        noise_var = noise_std ** 2
        def add_gaussian_noise(image, **kwargs):
            noise_std = 5.0 / 255.0
            noise = np.random.normal(0, noise_std, image.shape).astype(np.float32)
            return image + noise
        
        self.A_add_csnt = A.Compose([
            A.Lambda(image=add_gaussian_noise, p=1.0)
        ])


        self.albumentations_aug_ops = [
            self.A_do_nothing,
            self.A_flipud,
            self.A_rot90,
            self.A_rot90_flipud,
            self.A_rot180,
            self.A_rot180_flipud,
            self.A_rot270,
            self.A_rot270_flipud,
            self.A_add_csnt
        ]
        
        self.w_aug = [ # Original weights
            32, 12, 12, 12, 12, 12, 12, 12, 12,
        ]
        # Index for the noise operation (A_add_csnt) in self.albumentations_aug_ops
        self.noise_op_idx = 8


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
            img = self.transform(img) # self.transform is T.ToTensor()
            frames.append(img)

        try:
            if not all(isinstance(f, torch.Tensor) for f in frames):
                raise TypeError(
                    "All frames must be PyTorch Tensors before stacking. Ensure ToTensor() is in your transform."
                )
            # sequence_tensor is (F, C, H, W)
            sequence_tensor = torch.stack(frames)
        except Exception as e:
            print(
                f"Error stacking frames for sequence {idx} (paths: {sequence_paths}): {e}"
            )
            # TODO: Should this return None or raise error? Copilot: Returning None as per original structure.
            return None 

        cropped_sequence_tensor = sequence_tensor # Alias if no crop
        if self.crop_size:
            crop_h, crop_w = self.crop_size
            _, _, h, w = sequence_tensor.shape # F, C, H, W
            if h < crop_h or w < crop_w:
                # This case should ideally be handled by filtering sequences or resizing.
                # For now, raising an error as in original logic if crop is impossible.
                raise ValueError(
                    f"Crop size ({crop_h}, {crop_w}) is larger than image dimensions ({h}, {w})."
                )
            top = torch.randint(0, h - crop_h + 1, (1,)).item()
            left = torch.randint(0, w - crop_w + 1, (1,)).item()
            cropped_sequence_tensor = sequence_tensor[
                :, :, top : top + crop_h, left : left + crop_w
            ] # Shape: (F, C, H_cropped, W_cropped)

        
        # This view is for the final model input and GT extraction
        # (F*C, H_cropped, W_cropped)
        processed_sequence_for_gt_extraction = cropped_sequence_tensor.reshape(
            self.sequence_length * self.channels, cropped_sequence_tensor.size(-2), cropped_sequence_tensor.size(-1)
        )

        if self.apply_sequence_augmentations:
            # Albumentations operates on NumPy arrays (H, W, C)
            # Current cropped_sequence_tensor is (F, C, H_cropped, W_cropped)
            
            # Convert list of tensors (F, C, H, W) to list of numpy arrays (H, W, C)
            # Ensure tensor is on CPU before .numpy(). Input to ToTensor() was PIL.
            frames_to_augment_np = [
                frame_chw.permute(1, 2, 0).cpu().numpy() for frame_chw in cropped_sequence_tensor
            ]
            # frames_to_augment_np is a list of F frames, each (H_cropped, W_cropped, C)
            # Values are float32 in [0,1] range due to ToTensor()

            chosen_op_idx = random.choices(range(len(self.albumentations_aug_ops)), weights=self.w_aug, k=1)[0]
            chosen_op_pipeline = self.albumentations_aug_ops[chosen_op_idx]

            if chosen_op_idx == self.noise_op_idx: # Noise augmentation (A_add_csnt)
                # Apply noise independently to each frame
                transformed_frames_np = [
                    chosen_op_pipeline(image=frame_np)['image'] for frame_np in frames_to_augment_np
                ]
            else: # Geometric or DoNothing augmentation, apply consistently across frames
                  # chosen_op_pipeline is an A.Compose object. Its .transforms is the list of actual ops.
                  # This handles A_do_nothing correctly as its .transforms is empty.
                if not chosen_op_pipeline.transforms: # Empty list of transforms means A_do_nothing
                    transformed_frames_np = frames_to_augment_np
                else:
                    replay_pipeline = A.ReplayCompose(chosen_op_pipeline.transforms)
                    
                    # Apply to first frame
                    data = replay_pipeline(image=frames_to_augment_np[0])
                    transformed_frames_np = [data['image']]
                    # Replay for subsequent frames
                    for i in range(1, len(frames_to_augment_np)):
                        transformed_frames_np.append(
                            A.ReplayCompose.replay(data['replay'], image=frames_to_augment_np[i])['image']
                        )
            
            # Convert augmented NumPy frames (list of HWC) back to (F, C, H, W) tensor
            # Ensure data type consistency (albumentations should keep float32 if input is float32)
            transformed_frames_torch = [
                torch.from_numpy(frame_np.astype(np.float32)).permute(2, 0, 1) 
                for frame_np in transformed_frames_np
            ]
            augmented_sequence_tensor = torch.stack(transformed_frames_torch).to(cropped_sequence_tensor.device) # Move to original device
            
            # This is the actual augmented sequence to be used by the model
            # Reshape to (F*C, H_cropped, W_cropped)
            processed_sequence = augmented_sequence_tensor.reshape(
                self.sequence_length * self.channels, augmented_sequence_tensor.size(-2), augmented_sequence_tensor.size(-1)
            )
        else: # No augmentation
            processed_sequence = processed_sequence_for_gt_extraction


        # Extract GT frame using the original logic from the *final* processed_sequence
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
