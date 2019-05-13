import torch
import torch.nn as nn
import numpy as np


a = torch.randn(1, 100, 1, 1)

model = nn.Sequential(
    nn.ConvTranspose2d(100, 1024, 4, 1, 0, bias=False),
)

a = torch.Tensor(a)
print(a)
b = model(a)
print(b.size())
