
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


## 4. Initialise Shift-Net Submodule

The Shift-Net teacher model is included as a git submodule. If you cloned with `--recursive`, it's already there. Otherwise, run:

```bash
git submodule update --init --recursive
```


## 5. ShiftNet Teacher Model & Checkpoint

**For distillation training**, you need the pre-trained ShiftNet checkpoint:

```bash
# Option 1: Place your trained ShiftNet checkpoint in
./Shift-Net/net_denoise.pth

# Option 2: Update the config file
# Edit configs/distill.yaml and change:
teacher_checkpoint: "./path/to/your/checkpoint.pth"
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
