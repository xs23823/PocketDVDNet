"""Run the trained 5-frame PocketDVDNet model on a camera or video file."""

from __future__ import annotations

import argparse
import os
import select
import sys
import threading
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import torch
import torch.nn.functional as functional

from models.pocketdvdnet import PocketDVDnet

try:
    import termios
    import tty
except ImportError:  # Non-POSIX platforms (e.g. Windows) have no termios/tty.
    termios = None
    tty = None


ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / "pocket_models" / "pocketdvdnet5_mse_state.pt"
FRAME_COUNT = 5
WINDOW_TITLE = "Original (left) | PocketDVDNet (right)"


def is_quit_key(key: int) -> bool:
    """Recognize q, Q, and Escape from OpenCV's platform-specific key codes."""
    return key >= 0 and (key & 0xFF) in (ord("q"), ord("Q"), 27)


@contextmanager
def terminal_quit_listener() -> Iterator[threading.Event]:
    """Listen for q or Escape in a terminal without requiring Enter."""
    stopped = threading.Event()
    if termios is None or not sys.stdin.isatty():
        yield stopped
        return

    descriptor = sys.stdin.fileno()
    original_settings = termios.tcgetattr(descriptor)

    def listen() -> None:
        try:
            tty.setcbreak(descriptor)
            while not stopped.is_set():
                readable, _, _ = select.select([descriptor], [], [], 0.1)
                if readable and os.read(descriptor, 1) in (b"q", b"Q", b"\x1b"):
                    stopped.set()
        finally:
            termios.tcsetattr(descriptor, termios.TCSADRAIN, original_settings)

    listener = threading.Thread(target=listen, name="terminal-quit-listener", daemon=True)
    listener.start()
    try:
        yield stopped
    finally:
        stopped.set()
        listener.join(timeout=1.0)
        termios.tcsetattr(descriptor, termios.TCSADRAIN, original_settings)


def select_device(requested: str = "auto") -> torch.device:
    """Select CUDA, Apple Metal, or CPU in that order."""
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("Apple Metal (MPS) was requested but is not available")
        return device

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint_path: Path, device: torch.device) -> PocketDVDnet:
    """Load a tensor-only checkpoint without enabling arbitrary pickle objects."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run the local setup again or pass --checkpoint /path/to/weights.pt."
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model_state", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported checkpoint format in {checkpoint_path}")

    if state_dict and next(iter(state_dict)).startswith("module."):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}

    model = PocketDVDnet(num_input_frames=FRAME_COUNT, num_color_ch=3)
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def frame_to_tensor(frame_bgr: np.ndarray) -> torch.Tensor:
    """Convert an OpenCV BGR frame to a normalized RGB tensor."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(frame_rgb).permute(2, 0, 1).float().div_(255.0)


def resize_webcam_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Force webcam input to the requested processing size.

    OpenCV capture backends treat CAP_PROP_FRAME_WIDTH/HEIGHT as requests and
    macOS AVFoundation devices, especially Continuity Cameras, may ignore them.
    """
    if width <= 0 or height <= 0:
        raise ValueError("Webcam width and height must be positive")
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def denoise_frames(
    model: PocketDVDnet,
    frames: list[torch.Tensor],
    noise_sigma: float,
    device: torch.device,
) -> np.ndarray:
    """Denoise the centre image in a five-frame window."""
    if len(frames) != FRAME_COUNT:
        raise ValueError(f"Expected {FRAME_COUNT} frames, received {len(frames)}")

    height, width = frames[0].shape[-2:]
    if height < 4 or width < 4:
        raise ValueError(f"Frames must be at least 4x4 pixels, received {width}x{height}")
    if any(tuple(frame.shape) != tuple(frames[0].shape) for frame in frames):
        raise ValueError("All input frames must have the same dimensions")

    sequence = torch.cat(frames, dim=0).unsqueeze(0).to(device)
    pad_height = (-height) % 4
    pad_width = (-width) % 4
    if pad_height or pad_width:
        sequence = functional.pad(sequence, (0, pad_width, 0, pad_height), mode="reflect")

    padded_height, padded_width = sequence.shape[-2:]
    noise_map = torch.full(
        (1, FRAME_COUNT * 3, padded_height, padded_width),
        noise_sigma / 255.0,
        dtype=sequence.dtype,
        device=device,
    )

    with torch.inference_mode():
        output = model(sequence, noise_map).clamp_(0.0, 1.0)

    output = output[..., :height, :width].squeeze(0).cpu()
    output_rgb = output.permute(1, 2, 0).mul(255).byte().numpy()
    return cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)


def open_capture(source: str) -> tuple[cv2.VideoCapture, bool]:
    """Open a numeric camera index or a video path."""
    is_camera = source.isdecimal()
    capture_source: int | str = int(source) if is_camera else source
    capture = cv2.VideoCapture(capture_source)
    if not capture.isOpened():
        kind = "camera" if is_camera else "video"
        raise RuntimeError(f"Could not open {kind}: {source}")
    return capture, is_camera


def run(args: argparse.Namespace) -> None:
    device = select_device(args.device)
    model = load_model(args.checkpoint, device)
    capture, is_camera = open_capture(args.input)

    if is_camera:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(source_fps) or source_fps <= 1:
        source_fps = 30.0

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Input: {'camera ' if is_camera else ''}{args.input}")
    print(
        "Press q or Escape in the preview or terminal to stop; Ctrl-C also works."
        if not args.no_display
        else "Running without a display; press q, Escape, or Ctrl-C to stop."
    )

    frame_buffer: deque[torch.Tensor] = deque(maxlen=FRAME_COUNT)
    bgr_buffer: deque[np.ndarray] = deque(maxlen=FRAME_COUNT)
    writer: cv2.VideoWriter | None = None
    processed = 0
    captured = 0
    capture_seconds = 0.0
    denoise_seconds = 0.0
    started = time.perf_counter()
    reported_resolution = False

    if not args.no_display:
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)

    try:
        with terminal_quit_listener() as stopped:
            while not stopped.is_set():
                capture_started = time.perf_counter()
                ok, frame = capture.read()
                capture_seconds += time.perf_counter() - capture_started
                if not ok:
                    break
                captured += 1

                if not reported_resolution:
                    source_height, source_width = frame.shape[:2]
                    if is_camera:
                        print(f"Camera delivered: {source_width}x{source_height}")
                        print(f"Processing resolution: {args.width}x{args.height}")
                    else:
                        print(f"Processing resolution: {source_width}x{source_height}")
                    reported_resolution = True

                if is_camera:
                    frame = resize_webcam_frame(frame, args.width, args.height)

                frame_buffer.append(frame_to_tensor(frame))
                bgr_buffer.append(frame)
                if len(frame_buffer) < FRAME_COUNT:
                    continue

                denoise_started = time.perf_counter()
                denoised = denoise_frames(model, list(frame_buffer), args.noise_sigma, device)
                denoise_seconds += time.perf_counter() - denoise_started
                centre = bgr_buffer[FRAME_COUNT // 2]
                display_frame = np.hstack((centre, denoised))
                processed += 1

                if args.output is not None:
                    if writer is None:
                        args.output.parent.mkdir(parents=True, exist_ok=True)
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(
                            str(args.output), fourcc, source_fps, (display_frame.shape[1], display_frame.shape[0])
                        )
                        if not writer.isOpened():
                            raise RuntimeError(f"Could not create output video: {args.output}")
                    writer.write(display_frame)

                if not args.no_display:
                    elapsed = max(time.perf_counter() - started, 1e-9)
                    cv2.putText(
                        display_frame,
                        f"PocketDVDNet {processed / elapsed:.1f} FPS",
                        (16, 32),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (80, 255, 80),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow(WINDOW_TITLE, display_frame)
                    if is_quit_key(cv2.waitKeyEx(1)):
                        stopped.set()
                    elif cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                        stopped.set()

                if args.max_frames is not None and processed >= args.max_frames:
                    stopped.set()
    except KeyboardInterrupt:
        print("\nStopping PocketDVDNet...")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    elapsed = max(time.perf_counter() - started, 1e-9)
    print(f"End-to-end: {processed} frames in {elapsed:.2f}s ({processed / elapsed:.2f} FPS).")
    if captured and capture_seconds:
        print(f"Camera/video read: {capture_seconds / captured * 1000:.1f} ms per captured frame.")
    if processed and denoise_seconds:
        print(
            f"Model + tensor conversion: {denoise_seconds / processed * 1000:.1f} ms "
            f"per output frame ({processed / denoise_seconds:.2f} FPS)."
        )
    if args.output is not None:
        print(f"Saved comparison video to {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="0",
        help="camera index (default: 0) or path to a video file",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"trained 5-frame state dict (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--noise-sigma", type=float, default=30.0, help="expected noise level in 0-255 units")
    parser.add_argument("--width", type=int, default=1280, help="requested webcam width")
    parser.add_argument("--height", type=int, default=720, help="requested webcam height")
    parser.add_argument("--output", type=Path, help="optional MP4 comparison-video path")
    parser.add_argument("--no-display", action="store_true", help="disable the OpenCV preview window")
    parser.add_argument("--max-frames", type=int, help="stop after this many denoised frames")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
