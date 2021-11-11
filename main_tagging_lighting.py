# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 11:50:59 2021

@author: 15626
"""

import yaml
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from LitVAE import LitVAE
import torch.optim as optim
#from torchvision import datasets, transforms
from matplotlib import pyplot as plt
# import EarlyStopping
from pytorchtools import EarlyStopping
#from model_tagging import VAE
import argparse
import pandas as pd
from TaggingDataModule import TaggingDataModule
#from dataloader.Tagging_loader import Tagging_loader
import utils.io.image as io_func
from utils.sitk_np import np_to_sitk
import numpy as np
import shutil
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
import sys
sys.path.append('/usr/not-backed-up/scnc/MNIST-VAE-main/dataloader/Tagging_loader.py') 


# Ref: https://github.com/lyeoni/pytorch-mnist-VAE/blob/master/pytorch-mnist-VAE.ipynb
# Ref: https://towardsdatascience.com/building-a-convolutional-vae-in-pytorch-a0f54c947f71
# Ref: https://blog.floydhub.com/long-short-term-memory-from-zero-to-hero-with-pytorch/
# Ref: (To plot 100 result images) https://medium.com/the-data-science-publication/how-to-plot-mnist-digits-using-matplotlib-65a2e0cc068

# To solve Intel related matplotlib/torch error.
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

parser = argparse.ArgumentParser(description='Tagging/VAE')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='D:/lab/code/MNIST-VAE-main/configs/vae.yaml')
args = parser.parse_args()

with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)
        


os.makedirs("vae_images_tagging", exist_ok=True)
#os.makedirs("vae_test_tagging", exist_ok=True)
#os.makedirs("vae_images_cine")
#os.makedirs("vae_test_cine")

# For reproducibility
torch.manual_seed(config['logging_params']['manual_seed'])
np.random.seed(config['logging_params']['manual_seed'])
cudnn.deterministic = True
cudnn.benchmark = False

model = LitVAE()
trainer = pl.Trainer(gpus=0)
trainer.fit(model, train_dataloader, val_dataloader)
