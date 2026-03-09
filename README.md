# PocketDVDNet - Realtime Video Denoising for Real Camera Noise

A repository featuring **PocketDVDnet** (student models) trained with knowledge distillation using **ShiftNet** as teacher for efficient video denoising.

## Key Features

- **5-frame and 7-frame PocketDVDnet models** – Lightweight student architectures for real-time denoising
- **Knowledge Distillation option** – Train with teacher 
- **Realistic noise model** – Composite shot/read/uniform/row/periodic noise application

### File Structure

```
studfastdvdnet/
├── run.py                      # Standard training script (Charbonier loss)
├── distill.py                  # Knowledge distillation training script
├── trainer.py                  # Base Trainer class with core training logic
│
├── configs/                    # YAML configuration files
│   ├── charb.yaml              # Standard training (5-frame)
│   ├── distill.yaml            # Distillation training (5-frame)
│   └── obproxsg.yaml           # Sparsity pruning (5-frame)
│
├── models/                     # Model definitions
│   ├── pocketdvdnet.py         # PocketDVDnet 5-frame student
│   ├── pocketdvdnet7f.py       # PocketDVDnet7 7-frame student
│   ├── shiftnet.py             # ShiftNet teacher model
│   └── fastdvdnet/             # FastDVDnet full model
│       └── fastdvdnet.py
│
├── dataloaders/                # Data loading & preprocessing
│   ├── noise.py                # Composite noise model
│   ├── predicted_labels.csv    # Noise parameters
│   └── fastdvdnet/             # Dataset classes
│       ├── dataloader.py        # DVDDataset, ValDataset
│       └── utils.py             # batch_psnr, denoise utilities
│
├── eval/                       # Evaluation & inference scripts
│   ├── test_pocket.py          # Test on validation set
│   └── test_live.py            # Real-time inference on webcam
│
├── train_method/               # Optimization algorithms
│   └── obproxsg.py             # OBProxSG for sparsity
│
├── utils/                      # Utilities
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

This automatically clones the **Shift-Net** teacher model as a submodule into `Shift-Net/`. If you cloned without `--recursive`, run:

```bash
git submodule update --init --recursive
```

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

Paths are relative and should be set up according to your local structure (see Installation).

## Models

### Student Models

- **PocketDVDnet (5-frame)**: Lightweight 5-frame denoiser (in `student/pocketdvdnet.py`)
- **PocketDVDnet7 (7-frame)**: Lightweight 7-frame denoiser (in `student/pocketdvdnet7f.py`)
- **FastDVDnet (5-frame)**: Full 5-frame denoiser (in `student/fastdvdnet.py`)

### Teacher Model

- **ShiftNet**: Included as submodule in `Shift-Net/`; used for knowledge distillation training

## Noise Model

The composite noise model generates realistic video noise by combining:
- **Shot noise** (Poisson): Sensor photon noise
- **Read noise** (Gaussian): Sensor electronics noise
- **Uniform noise**: Quantization and dithering
- **Row noise**: Row-dependent noise patterns
- **Periodic noise**: Periodic artifacts

Configured via `dataloaders/noise.py` with parameters from config files.
