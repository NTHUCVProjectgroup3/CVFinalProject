import torch
import numpy as np

G = torch.load('./TrainedModels/cityscapes/scale_factor=0.750000_seg/0/netG.pth')
print(np.abs(G['head.conv.weight'].cpu().numpy()).mean(axis=(0,2,3)))