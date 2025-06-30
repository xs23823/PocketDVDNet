import torch
import numpy as np
import pandas as pd
import random

class NoiseModel:
    """
    Applies a composite noise model to an input image tensor by adding several types
    of noise, including shot/read noise, uniform noise, row noise, row temporal noise,
    and periodic noise.
    Handles input shape [B, C*F, H, W] with F=7, ensuring noise consistency across F frames.
    """

    def __init__(self, dict_path, seed=-1):
        """
        Args:
            dict_path (str): Path to CSV with noise parameters.
            seed (int): Random seed for deterministic noise. If -1 or None, uses random seed.
        """
        self.seed = seed
        self.actual_labels = {
            "shot_noise": [0, 0.5],
            "read_noise": [0, 0.1],
            "uniform_noise": [0, 0.1],
            "row_noise": [0, 0.01],
            "row_noise_temp": [0, 0.01],
            "periodic0": [0, 0.5],
            "periodic1": [0, 0.5],
            "periodic2": [0, 0.5],
        }

        try:
            self.labels = pd.read_csv(dict_path)
        except FileNotFoundError:
            print(f"Warning: {dict_path} not found")
            raise RuntimeError("Cannot get new noise parameters.")

    def get_noise_parameters(self):
        """
        Samples a new set of unscaled noise parameters from the loaded CSV.
        Returns:
            dict: A dictionary of unscaled noise parameters.
        """
        if self.seed is None or self.seed == -1:
            sample_seed = random.randint(0, 2**32 - 1)
        else:
            sample_seed = self.seed
        return self.labels.sample(n=1, random_state=sample_seed).iloc[0].to_dict()

    def _scale_noise_dict(self, noise_dict_not_scaled):
        """
        Scales the values in a noise dictionary to the actual label ranges.

        Given a dictionary of unscaled noise values (`noise_dict_not_scaled`), this method scales each value to the corresponding range defined in `self.actual_labels`. If a key from `self.actual_labels` is missing in the input dictionary, a warning is printed and the minimum value of the range is used as the default.

        Args:
            noise_dict_not_scaled (dict): Dictionary containing unscaled noise values, keyed by label.

        Returns:
            dict: Dictionary with noise values scaled to the ranges specified in `self.actual_labels`.
        """

        noise_dict = {}
        for key in self.actual_labels:
            if key in noise_dict_not_scaled:
                vmin, vmax = self.actual_labels[key]
                scale = vmax - vmin
                noise_dict[key] = scale * noise_dict_not_scaled[key] + vmin
            else:
                print(f"Warning: Key '{key}' not in noise_dict_not_scaled. Defaulting its scaled value.")
                noise_dict[key] = self.actual_labels[key][0]
        return noise_dict

    def __call__(self, x):
        """
        Applies the noise transformation to the input tensor.
        Args:
            x (torch.Tensor): The input tensor to which the noise will be applied.
        Returns:
            torch.Tensor: The transformed tensor with noise applied.
        """
        return self.apply(x)

    def apply(self, x):
        """
        Applies a combination of synthetic noise types to the input tensor `x`.

        The function simulates various noise sources commonly encountered in imaging pipelines, including:
            1. Shot and read noise (signal-dependent and signal-independent Gaussian noise)
            2. Uniform noise (random uniform noise)
            3. Row noise (Gaussian noise consistent across each row)
            4. Row temporal noise (Gaussian noise consistent across each row and frame)
            5. Periodic noise (low-frequency periodic noise generated in the frequency domain)

        The noise parameters are determined by `self.get_noise_parameters()` and scaled by `self._scale_noise_dict`.
        The function expects the input tensor `x` to be in the shape [B, C*F, H, W], where:
            - B: Batch size
            - C: Number of channels
            - F: Number of frames (fixed at 5)
            - H: Height
            - W: Width

        The function ensures deterministic or random behavior based on `self.seed`.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C*F, H, W], with values in [0, 1].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - The noisy tensor of shape [B, C*F, H, W], clipped to [0, 1].
                - The total added noise tensor of shape [B, C*F, H, W].
        """

        # Set seeds for deterministic behavior, or random if seed is -1/None
        if self.seed is None or self.seed == -1:
            torch_seed = random.randint(0, 2**32 - 1)
            np_seed = random.randint(0, 2**32 - 1)
        else:
            torch_seed = self.seed
            np_seed = self.seed

        torch.manual_seed(torch_seed)
        np.random.seed(np_seed)

        self.device = x.device
        self.noise_dict = self.get_noise_parameters()
        self.noise_dict = self._scale_noise_dict(self.noise_dict)
        B, CF, H, W = x.shape
        F_frames = 7 #changed for new architecture, was 5

        if CF % F_frames != 0:
            raise ValueError(f"Input channel-frame dimension ({CF}) is not divisible by F_frames ({F_frames}).")
        C_channels = CF // F_frames

        assert x.min() >= 0 and x.max() <= 1.0005, "Input tensor should be in [0, 1] range"

        # Reshape x to [B, C, F, H, W] for frame-consistent noise application
        x_view = x.view(B, C_channels, F_frames, H, W)

        total_added_noise = torch.zeros_like(x_view).to(self.device)

        # 1. Shot and Read Noise
        factor_sr_per_pixel = x_view * self.noise_dict["shot_noise"] + self.noise_dict["read_noise"]
        randn_base_sr = torch.randn(B, C_channels, 1, H, W, device=self.device)
        randn_expanded_sr = randn_base_sr.expand(B, C_channels, F_frames, H, W)
        current_shot_read_noise = randn_expanded_sr * factor_sr_per_pixel
        total_added_noise += current_shot_read_noise

        # 2. Uniform Noise
        rand_base_uniform = torch.rand(B, C_channels, 1, H, W, device=self.device)
        rand_expanded_uniform = rand_base_uniform.expand(B, C_channels, F_frames, H, W)
        current_uniform_noise = self.noise_dict["uniform_noise"] * rand_expanded_uniform
        total_added_noise += current_uniform_noise

        # 3. Row Noise
        randn_base_row = torch.randn(B, C_channels, 1, H, 1, device=self.device)
        randn_expanded_row = randn_base_row.expand(B, C_channels, F_frames, H, W)
        current_row_noise = self.noise_dict["row_noise"] * randn_expanded_row
        total_added_noise += current_row_noise

        # 4. Row Temporal Noise
        randn_base_row_temp = torch.randn(B, 1, 1, 1, W, device=self.device)
        randn_expanded_row_temp = randn_base_row_temp.expand(B, C_channels, F_frames, H, W)
        current_row_temp_noise = self.noise_dict["row_noise_temp"] * randn_expanded_row_temp
        total_added_noise += current_row_temp_noise

        # 5. Periodic Noise
        periodic_noise_fft_base = torch.zeros(B, C_channels, 1, H, W, dtype=torch.cfloat, device=self.device)

        p0_val = self.noise_dict["periodic0"] * torch.randn(B, C_channels, device=self.device)
        p1_val = self.noise_dict["periodic1"] * torch.randn(B, C_channels, device=self.device)
        p2_val = self.noise_dict["periodic2"] * torch.randn(B, C_channels, device=self.device)

        periodic_noise_fft_base[:, :, 0, 0, 0] = p0_val
        complex_val_slice = torch.complex(p1_val, p2_val)
        periodic_noise_fft_base[:, :, 0, 0, W // 4] = complex_val_slice
        periodic_noise_fft_base[:, :, 0, 0, 3 * W // 4] = torch.complex(p1_val, -p2_val)

        periodic_gen_base_spatial = torch.abs(torch.fft.ifft2(periodic_noise_fft_base, dim=(-2, -1), norm="ortho"))
        current_periodic_noise = periodic_gen_base_spatial.expand(B, C_channels, F_frames, H, W)
        total_added_noise += current_periodic_noise

        # Add combined noise to image and clip
        noisy_view = x_view + total_added_noise
        noisy_view_clipped = torch.clip(noisy_view, 0, 1)

        # Reshape back to [B, C*F, H, W]
        noisy_final = noisy_view_clipped.view(B, CF, H, W)
        total_added_noise_reshaped = total_added_noise.view(B, CF, H, W)

        return noisy_final, total_added_noise_reshaped
