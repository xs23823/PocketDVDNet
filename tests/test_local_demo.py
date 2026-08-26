from pathlib import Path

import numpy as np
import torch

from local_demo import (
    DEFAULT_CHECKPOINT,
    denoise_frames,
    is_quit_key,
    load_model,
    resize_webcam_frame,
)


def test_quit_keys() -> None:
    assert is_quit_key(ord("q"))
    assert is_quit_key(ord("Q"))
    assert is_quit_key(27)
    assert is_quit_key(0x100000 + ord("q"))
    assert not is_quit_key(ord("x"))


def test_webcam_frame_is_forced_to_processing_resolution() -> None:
    native_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    resized = resize_webcam_frame(native_frame, width=640, height=360)

    assert resized.shape == (360, 640, 3)


def test_trained_model_denoises_five_frame_window() -> None:
    assert Path(DEFAULT_CHECKPOINT).is_file()
    model = load_model(DEFAULT_CHECKPOINT, torch.device("cpu"))
    frames = [torch.rand(3, 33, 35) for _ in range(5)]

    output = denoise_frames(model, frames, noise_sigma=30.0, device=torch.device("cpu"))

    assert output.shape == (33, 35, 3)
    assert output.dtype == np.uint8
