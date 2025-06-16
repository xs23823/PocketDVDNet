from torchinfo import summary
import torch
import torch.nn as nn
from models.fastdvdnet import FastDVDnet
from models.pocketdvdnet import PocketDVDnet

model = FastDVDnet()
batch_size = 16
summary(model)

model = PocketDVDnet()
batch_size = 16
summary(model)

# Test your system's pin_memory tolerance

import torch
data = torch.randn(64, 15, 96, 96)
loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(data), 
    batch_size=128, 
    pin_memory=True
)
for batch in loader:
    print('Pin memory works!')
    break


