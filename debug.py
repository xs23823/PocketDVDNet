from torchinfo import summary
import torch
import torch.nn as nn
from student import PocketDVDnet7
from student import FastDVDnet
#from teachers.EDVR.archs.edvr_arch import EDVR
import argparse

model = PocketDVDnet7()
batch_size = 16
summary(model)


# Setup dummy input and noise maps
batch_size = 1
num_frames = 7
num_color_ch = 3
height, width = 64, 64  # example spatial size
noise_ch_per_frame = 3  # must match what model expects

# Input tensor shape: (batch, channels, height, width)
# channels = num_frames * num_color_ch
x = torch.randn(batch_size, num_frames * num_color_ch, height, width)

# Noise map bundle shape: batch, noise_channels, height, width
# noise_channels = num_frames * noise_ch_per_frame
noise_map_bundle = torch.randn(batch_size, num_frames * noise_ch_per_frame, height, width)

# Instantiate model
model = PocketDVDnet7(num_input_frames=num_frames, num_color_ch=num_color_ch, noise_ch_per_frame=noise_ch_per_frame)

# Forward pass
output = model(x, noise_map_bundle)
print(model)

print(f'Output shape: {output.shape}')


   
