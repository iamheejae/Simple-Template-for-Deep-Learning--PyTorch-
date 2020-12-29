from __future__ import division
import os
import numpy as np
import h5py
import torch
import torch.nn.functional as F
import pdb 
import scipy.misc
from random import *
from torch.utils.data import Dataset

class mydataset(Dataset):

    def __init__(self, root, option):       
        path = os.path.join(root)  #.h5 dataset path
        f = h5py.File(path, 'r')

        self.option = option     
        self.group1 = f.get('/group1')
        self.group1_dataset = np.array(self.group1, dtype = np.float32)

    def __len__(self):
        return self.group1_dataset.shape[0]

    def GenerateOptionPair(self, option):

        if (option == 'random'):         
            option1 = np.random.randint(1,100)
            option2 = np.random.randint(1,100)

        else:
            raise Exception('Wrong option')    

        return option1, option2

    def __getitem__(self, idx):      
 
        option1, option2 = self.GenerateOptionPair(self.option)

        input1 = self.group1_dataset[idx, option1, :, :, :] # target
        input2 = self.group1_dataset[idx, option2, :, :, :] # source
        gt = self.group1_dataset[idx, option1, :, :, :] # gt
    
        # transform np image to torch tensor
        gt_tensor = torch.Tensor(img_gt)
        input1_tensor = torch.Tensor(input1)        
        input2_tensor = torch.Tensor(input2)              
      
        return gt_tensor, input1_tensor, input2_tensor

