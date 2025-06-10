from torchinfo import summary
import torch
import torch.nn as nn
from models.fastdvdnet import FastDVDnet

model = FastDVDnet()
batch_size = 16
summary(model)