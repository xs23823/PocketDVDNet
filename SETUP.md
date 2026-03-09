
## System Requirements

- **Python**: 3.7 or higher 

### Dataset Structure

```
├── data/
│   ├── DAVIS_val/
│   │   ├── aerialtown/
│   │   ├── bear/
│   │   └── [other sequences]
│   └── train/
│       ├── sequence1/
│       ├── sequence2/
│       └── [other sequences]
└── results/
    ├── charb/
    ├── distill/
    └── [other output dirs]
```


## 4. Initialize Shift-Net Submodule

The Shift-Net teacher model is included as a git submodule. If you cloned with `--recursive`, it's already initialized. Otherwise, run:

```bash
git submodule update --init --recursive
```

This clones the ShiftNet repository into `Shift-Net/` at the repo root.

## 5. ShiftNet Teacher Model & Checkpoint

**For distillation training**, you need the pre-trained ShiftNet checkpoint:

```bash
# Option 1: Place your trained ShiftNet checkpoint in
./Shift-Net/net_denoise.pth

# Option 2: Update the config file
# Edit configs/distill.yaml and change:
teacher_checkpoint: "./path/to/your/checkpoint.pth"
```

If you don't have a pre-trained ShiftNet checkpoint:
1. Train ShiftNet separately (see [Shift-Net repository](https://github.com/dasongli1/Shift-Net))
2. Use standard training (`run.py`) instead of distillation

## 6. Configuration Files

Config files in `configs/` use relative paths:

```yaml
trainset_dir: "./data/train"          # Training dataset
valset_dir: "./data/DAVIS_val"        # Validation dataset
out_dir: "./results/charb"            # Output checkpoints
teacher_checkpoint: "./Shift-Net/net_denoise.pth"  # Teacher model (distill only)
```

## Quick Start

```bash
# Standard training 
python run.py --config configs/charb.yaml 

# Distillation training
python distill.py --config configs/distill.yaml

# Evaluation
python eval/test_pocket.py --config configs/charb.yaml --model_path results/charb/model.pth --num__input_frames 5 --noise_sigma 30
```
