# PocketDVDNet - Realtime Video Denoising for Real Camera Noise

- **5-frame and 7-frame PocketDVDnet models** – Lightweight student architecture.
- Knowledge Distillation option with shiftnet teacher.
- Realistic noise model

## Local demo

This checkout includes a macOS/Linux-friendly runner for the trained 5-frame
model. It automatically uses CUDA, Apple Metal (MPS), or CPU, in that order.

```bash
# Install (Python 3.12 is recommended)
UV_CACHE_DIR=.uv-cache uv sync --python 3.12

# Live camera: original on the left, denoised output on the right
.venv/bin/python local_demo.py

# Reduce the actual processing resolution for a faster preview
.venv/bin/python local_demo.py --width 640 --height 360

# Process a video without opening a window
.venv/bin/python local_demo.py \
  --input /path/to/input.mp4 \
  --output results/comparison.mp4 \
  --no-display
```

Press `q` or Escape in either the preview window or the terminal to stop. Closing
the preview window and pressing Ctrl-C are also supported.

On macOS, camera resolution settings are only hints and may be ignored by
AVFoundation or Continuity Camera. The local runner therefore resizes webcam
frames explicitly and prints both the camera-delivered and processing sizes.

The local checkpoint is expected at
`pocket_models/pocketdvdnet5_mse_state.pt`. The upstream repository removed the
trained PocketDVDNet checkpoints from its latest tree; this local setup restores
the matching 5-frame checkpoint from the repository's history and converts it to
a tensor-only state dictionary for safe loading.

### iOS demo

The checkout also includes an iPhone app that runs the trained model on-device
with a live side-by-side original/denoised camera view. See
[ios/README.md](ios/README.md) for build instructions. The model is converted
from the PyTorch checkpoint with `convert_to_coreml.py` (needs the `coreml`
dependency group, which adds `coremltools` and `pillow`):

```bash
UV_CACHE_DIR=.uv-cache uv sync --python 3.12 --group coreml
.venv/bin/python convert_to_coreml.py --width 640 --height 360 --noise-sigma 30
```

### File Structure

```
Pocketdvdnet/
├── run.py                      # Standard training script (Charbonier loss)
├── distill.py                  # Knowledge distillation training script
├── trainer.py                  # Base Trainer class with core training logic
│
├── configs/                    # YAML configuration files
│   ├── charb.yaml              # Standard training (5-frame)
│   ├── distill.yaml            # Distillation training (5-frame)
│   └── obproxsg.yaml           # Sparsity pruning (5-frame)
│
├── models/                     
│   ├── pocketdvdnet.py         # PocketDVDnet 5-frame student
│   ├── pocketdvdnet7f.py       # PocketDVDnet7 7-frame student
│   ├── shiftnet.py             # ShiftNet teacher model
│   └── fastdvdnet/             # FastDVDnet full model
│       └── fastdvdnet.py
│
├── dataloaders/               
│   ├── noise.py                # Composite noise model
│   ├── predicted_labels.csv    # Noise parameters
│   └── fastdvdnet/             # Dataset classes
│       ├── dataloader.py        # DVDDataset, ValDataset
│       └── utils.py             # batch_psnr, denoise utilities
│
├── eval/                       
│   ├── test_pocket.py          # Test on validation set
│   └── test_live.py            # Real-time inference on webcam
│
├── train_method/               # Optimisation algorithms
│   └── obproxsg.py             # OBProxSG for sparsity
│
├── utils/                    
│   ├── prefetcher.py           # Data prefetching
│   └── lr_scheduler.py         # Learning rate schedulers
│
├── Shift-Net/                  # Teacher model (submodule)
├── results/                    # Training outputs (auto-created)
├── data/                       # Datasets (auto-created, see SETUP.md)
│
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── SETUP.md                    # Installation & setup guide
=
```

## Installation

### 1. Clone with Submodules

```bash
git clone --recursive https://github.com/xs23823/PocketDVDnet.git
```

This automatically clones the **Shift-Net** teacher model as a submodule.

### 2. Install Dependencies

See [SETUP.md](SETUP.md) for setup instructions.


## Configuration

All training is controlled via YAML config files in `configs/`. Key parameters:

- **sequence_length**: Number of input frames (5 or 7)
- **batch_size**: Batch size for training
- **lr**: Initial learning rate
- **iterations**: Total training iterations
- **trainset_dir**: Path to training dataset
- **valset_dir**: Path to validation dataset
- **out_dir**: Output directory for checkpoints and logs


## Models

### Student Models

- **PocketDVDnet (5-frame)**: Lightweight 5-frame denoiser (in `student/pocketdvdnet.py`)
- **PocketDVDnet7 (7-frame)**: Lightweight 7-frame denoiser (in `student/pocketdvdnet7f.py`)
- **FastDVDnet (5-frame)**: Full 5-frame denoiser (in `student/fastdvdnet.py`)

### Teacher Model

- **ShiftNet**: Included as submodule in `Shift-Net/`; used for knowledge distillation training

## Noise Model

The composite noise model generates realistic video noise by combining:
- **Shot noise** 
- **Read noise** 
- **Uniform noise**:
- **Row noise**:
- **Periodic noise**:
  
Configured via `dataloaders/noise.py` 
