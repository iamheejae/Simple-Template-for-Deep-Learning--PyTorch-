import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import sys
import pdb
import os
import cv2

from model_utils import *

class main_model(nn.Module):
    
    def __init__(self):
        super(main_model, self).__init__()

        """
        Define sub modules which are defined in /models/model_utils.py or /models
        ex. module1 = self.module1()
        """

    def forward(self, input1, input2):
        B, C, H, W = input1.size()
 

        """
        Final output
        ex. final_output = self.module1(input1, input2)
        """ 

        net_output = {'out' : final_output}  # output as dictionary

        return net_output


