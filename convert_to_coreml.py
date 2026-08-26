"""Convert the trained 5-frame PocketDVDNet checkpoint to a Core ML model.

The noise sigma is baked into the graph so the Core ML model has a single
logical input concept (five RGB frames) and one output image. Re-run this
script with a different --noise-sigma to rebuild for another noise level.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn.functional as functional

from local_demo import FRAME_COUNT, load_model
from models.pocketdvdnet import PocketDVDnet


ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / "pocket_models" / "pocketdvdnet5_mse_state.pt"
DEFAULT_OUTPUT = ROOT / "pocket_models" / "PocketDVDNet.mlpackage"
FRAME_NAMES = [f"frame{index}" for index in range(FRAME_COUNT)]


class CoreMLWrapper(torch.nn.Module):
    """Five RGB frames in [0, 1] -> denoised centre frame in [0, 1].

    The noise map is built from a fixed sigma and the input is padded to a
    multiple of 4 (then cropped back) so any fixed conversion size works.
    """

    def __init__(self, model: PocketDVDnet, noise_sigma: float, height: int, width: int):
        super().__init__()
        self.model = model
        self.noise_sigma = noise_sigma
        self.height = height
        self.width = width

    def forward(self, frame0, frame1, frame2, frame3, frame4):
        frames = (frame0, frame1, frame2, frame3, frame4)
        sequence = torch.cat(frames, dim=1)
        height, width = self.height, self.width
        pad_height = (-height) % 4
        pad_width = (-width) % 4
        if pad_height or pad_width:
            sequence = functional.pad(sequence, (0, pad_width, 0, pad_height), mode="replicate")

        noise_map = torch.full(
            (1, FRAME_COUNT * 3, height + pad_height, width + pad_width),
            self.noise_sigma / 255.0,
            dtype=sequence.dtype,
        )
        output = self.model(sequence, noise_map).clamp_(0.0, 1.0)
        # Output ImageType pixels are used as-is (scale must be 1.0), so emit 0-255.
        return output[..., :height, :width] * 255.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"trained 5-frame state dict (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Core ML model destination (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument("--noise-sigma", type=float, default=30.0, help="noise level in 0-255 units baked into the model")
    parser.add_argument("--width", type=int, default=640, help="input width the model is compiled for")
    parser.add_argument("--height", type=int, default=360, help="input height the model is compiled for")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")
    model = load_model(args.checkpoint, device)

    wrapper = CoreMLWrapper(model, args.noise_sigma, args.height, args.width).eval()
    example_inputs = tuple(torch.rand(1, 3, args.height, args.width) for _ in range(FRAME_COUNT))
    traced = torch.jit.trace(wrapper, example_inputs)

    inputs = [
        ct.ImageType(
            name=name,
            shape=(1, 3, args.height, args.width),
            color_layout=ct.colorlayout.RGB,
            scale=1.0 / 255.0,
        )
        for name in FRAME_NAMES
    ]
    outputs = [
        ct.ImageType(
            name="denoised",
            color_layout=ct.colorlayout.RGB,
        )
    ]

    coreml_model = ct.convert(
        traced,
        inputs=inputs,
        outputs=outputs,
        minimum_deployment_target=ct.target.iOS16,
        compute_units=ct.ComputeUnit.ALL,
    )
    coreml_model.short_description = (
        f"PocketDVDNet 5-frame video denoiser (noise sigma {args.noise_sigma:g}). "
        f"Inputs: five consecutive RGB frames; output: denoised centre frame."
    )
    for index, name in enumerate(FRAME_NAMES):
        coreml_model.input_description[name] = f"frame {index} of 5 (chronological, centre is frame 2)"
    coreml_model.output_description["denoised"] = "denoised version of the centre input frame"

    coreml_model.save(str(args.output))
    print(f"Saved {args.output} ({args.width}x{args.height}, sigma {args.noise_sigma:g})")

    from PIL import Image

    with torch.inference_mode():
        expected = traced(*example_inputs).numpy()
    pil_inputs = {
        name: Image.fromarray((frame.numpy()[0].transpose(1, 2, 0) * 255).astype(np.uint8), mode="RGB")
        for name, frame in zip(FRAME_NAMES, example_inputs)
    }
    prediction = coreml_model.predict(pil_inputs)
    predicted_image = prediction["denoised"]
    predicted = np.asarray(predicted_image, dtype=np.float32)[..., :3].transpose(2, 0, 1)[None]
    max_abs_diff = float(np.abs(expected - predicted).max())
    print(f"Max abs difference vs PyTorch: {max_abs_diff:.3f} / 255")
    if max_abs_diff > 3.0:
        raise SystemExit("Conversion verification failed: outputs differ too much from PyTorch")
    print("Verification passed (differences are within uint8 output quantization).")


if __name__ == "__main__":
    main()
