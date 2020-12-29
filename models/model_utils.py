import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable, Function
import pdb
import cv2


# convolutions
def conv(in_channel, out_channel, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=True),
            nn.LeakyReLU(0.1)
            )

def deconv(in_channel, out_channel, kernel_size=3, stride=2, padding=1):
    return nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride, padding, bias=True)



