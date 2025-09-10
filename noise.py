import torch
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, Union
from pathlib import Path


class NoiseModel:
    """
    Applies a composite noise model to an input image tensor.
    Input shape: [B, C*F, H, W] where F (frames) defaults to 5.

    Behavior:
      * If a CSV path is provided and valid, parameters are sampled by rows (normalized 0–1 then rescaled).
      * If no CSV is provided (csv_path is None or empty, or file not found), parameters are drawn uniformly
        and independently from each range in self.param_ranges every call (random combinations).
    """

    def __init__(self, csv_path: Optional[Union[str, Path]] = None, seed: Optional[int] = None, frames: int = 7):
        """
        Args:
            csv_path: Path to CSV with normalized noise parameters (0–1). If None or missing, sample uniformly.
            seed: Optional base random seed. If None, a new random seed is used per sampling.
            frames: Number of frames per sample (layout [B, C*F, H, W]).
        """
        self.seed = seed
        self.frames = frames
        self.device = torch.device("cpu")  # updated at call-time

        self.param_ranges = {
            "shot_noise": (0.0, 0.5),
            "read_noise": (0.0, 0.1),
            "uniform_noise": (0.0, 0.05),
            "row_noise": (0.0, 0.01),
            "row_temp_noise": (0.0, 0.01),
            "periodic_amplitude": (0.0, 0.2),
        }

        self.use_csv = False
        self.param_df = None

        if csv_path:
            try:
                path_obj = Path(csv_path)
                if path_obj.is_file():
                    self.param_df = pd.read_csv(path_obj)
                    if self.param_df.empty:
                        raise ValueError(f"Noise parameter CSV '{csv_path}' is empty.")
                    # Ensure all expected keys are present
                    for key in self.param_ranges:
                        if key not in self.param_df.columns:
                            self.param_df[key] = 0.0
                    self.use_csv = True
                else:
                    print(f"NoiseModel: CSV '{csv_path}' not found. Falling back to random uniform sampling.")
            except Exception as e:
                print(f"NoiseModel: Failed to load CSV '{csv_path}' ({e}). Using random uniform sampling.")

        if not self.use_csv:
            # Create a dummy DataFrame structure just for reference (optional)
            self.param_df = None

    def _get_random_params(self) -> Dict[str, float]:
        """Samples parameters either from CSV (row) or uniform ranges if no CSV."""
        seed = self.seed if self.seed is not None else np.random.randint(0, 1 << 30)
        np.random.seed(seed)
        if self.use_csv and self.param_df is not None:
            row = self.param_df.sample(n=1, random_state=seed).iloc[0]
            return {
                key: (self.param_ranges[key][1] - self.param_ranges[key][0]) * float(row[key]) + self.param_ranges[key][0]
                for key in self.param_ranges
            }
        else:
            # Independent uniform sampling within each range
            return {
                key: np.random.uniform(low=rng[0], high=rng[1])
                for key, rng in self.param_ranges.items()
            }

    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.apply(x)

    def apply(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply various types of noise to the input tensor.
        This method simulates realistic camera sensor noise by applying multiple
        noise types including shot noise, read noise, uniform noise, row noise,
        row temporal noise, and periodic noise.
        Args:
            x (torch.Tensor): Input tensor of shape (B, CF, H, W) where:
                - B: batch size
                - CF: channels × frames (must be divisible by self.frames)
                - H: height
                - W: width
                Values must be normalized to [0, 1] range.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - Noisy tensor: Input tensor with applied noise, clamped to [0, 1]
                - Noise map: Tensor of shape (B, F, 1, H, W) representing the effective noise standard deviation
        Raises:
            AssertionError: If input values are not in [0, 1] range or if CF is not
                           divisible by self.frames.
        Note:
            The method applies noise in the following order:
            1. Shot noise (Poisson-distributed, intensity-dependent)
            2. Read noise (Gaussian-distributed)
            3. Uniform noise (uniformly distributed)
            4. Row noise (correlated across columns)
            5. Row temporal noise (correlated across time)
            6. Periodic noise (frequency domain artifacts)
        """

        assert x.min() >= 0 and x.max() <= 1.0, "Input must be normalized to [0, 1]"      
        if x.dim() == 4:
            B, FC, H, W = x.shape
            assert FC % self.frames == 0, f"Input channels ({FC}) must be divisible by frames ({self.frames})"
            C = 3
            F = self.frames
        elif x.dim() == 5:
            B, F, C, H, W = x.shape
        else:
            raise ValueError("Input tensor must be 4D or 5D")
        

        x = x.reshape(B, C, F, H, W)
        self.device = x.device

        params = self._get_random_params()
        noise = torch.zeros_like(x)

        # 1. Shot Noise (Poisson)
        shot_factor = params["shot_noise"]
        if shot_factor > 0:
            scale = 255.0
            x_scaled = torch.clip(x * scale, 0, None)
            poisson_noise = torch.poisson(x_scaled) / scale - x
            noise += shot_factor * poisson_noise

        # 2. Read Noise (Gaussian)
        if params["read_noise"] > 0:
            noise += params["read_noise"] * torch.randn_like(x)

        # 3. Uniform Noise
        if params["uniform_noise"] > 0:
            noise += params["uniform_noise"] * (torch.rand_like(x) - 0.5)

        # 4. Row Noise
        if params["row_noise"] > 0:
            row = torch.randn(B, C, 1, H, 1, device=self.device)
            noise += params["row_noise"] * row

        # 5. Row Temporal Noise
        if params["row_temp_noise"] > 0:
            row_temp = torch.randn(B, 1, 1, 1, W, device=self.device)
            noise += params["row_temp_noise"] * row_temp

        # 6. Periodic Noise
        if params["periodic_amplitude"] > 0:
            freq_img = torch.zeros(B, C, 1, H, W, dtype=torch.cfloat, device=self.device)
            amp = params["periodic_amplitude"]
            freqs = [W // 4, W // 2, 3 * W // 4]
            for f in freqs:
                phase = 2 * torch.pi * torch.rand(B, C, device=self.device)
                freq_img[:, :, 0, 0, f] = amp * torch.exp(1j * phase)
                freq_img[:, :, 0, 0, -f] = torch.conj(freq_img[:, :, 0, 0, f])
            periodic = torch.fft.ifft2(freq_img, dim=(-2, -1), norm="ortho").real.abs()
            noise += periodic.expand(-1, -1, F, -1, -1)

        x_noisy = torch.clamp(x + noise, 0, 1)
        
        # Create noise map - represents the combined noise standard deviation
        # Calculate effective noise standard deviation across all noise types
        effective_noise_std = np.sqrt(
            params["shot_noise"]**2 + 
            params["read_noise"]**2 + 
            params["uniform_noise"]**2/12 +  # variance of uniform distribution is std^2/12
            params["row_noise"]**2 + 
            params["row_temp_noise"]**2 + 
            params["periodic_amplitude"]**2/2  # approximate variance for periodic noise
        )
        
        # Create noise map with shape (B, F, 1, H, W) to match the old model format
        noise_map = torch.full((B, F, 1, H, W), effective_noise_std, device=self.device)
        
        return x_noisy.view(B, F, C, H, W), noise_map
