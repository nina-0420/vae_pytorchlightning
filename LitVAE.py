# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 11:06:29 2021

@author: 15626
"""

import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from TaggingDataModule import TaggingDataModule

class LitVAE(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.encoder =torch.nn.Sequential(
        torch.nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),#64*64
        #torch.nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),#14*14
        torch.nn.BatchNorm2d(32),
        torch.nn.LeakyReLU(0.2, inplace=True),
        
        torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),#32*32
        torch.nn.BatchNorm2d(64),
        torch.nn.LeakyReLU(0.2, inplace=True),

        torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),#32*32
        torch.nn.BatchNorm2d(128),
        torch.nn.LeakyReLU(0.2, inplace=True),        


        torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),#32*32
        torch.nn.BatchNorm2d(128),
        torch.nn.LeakyReLU(0.2, inplace=True),
        )

        self.get_mu=torch.nn.Sequential(
            torch.nn.Linear(128 * 16 * 16, 512)#32
            #torch.nn.Linear(128 * 7 * 7, 1)#32
        )
        self.get_logvar = torch.nn.Sequential(
            torch.nn.Linear(128 * 16 * 16, 512)#32
        )
        self.get_temp = torch.nn.Sequential(
            torch.nn.Linear(512, 128 * 32 * 32)#32
        )
        self.decoder = torch.nn.Sequential(
                       
        torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
        torch.nn.ReLU(inplace=True),
        
        torch.nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
        torch.nn.ReLU(inplace=True),

        torch.nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1),   
        torch.nn.Sigmoid(),
        #torch.nn.Tanh(),
        )
# =============================================================================
#         super().__init__()
#         self.encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
#         self.decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))
# =============================================================================

    def get_z(self,mu,logvar):
        eps=torch.randn(mu.size(0),mu.size(1))#64,32
        eps=torch.autograd.Variable(eps)
        if torch.cuda.is_available():
           cuda = True
           device = 'cuda'
        else:
           cuda = False
           device = 'cpu'
        eps=eps.to(device)
        z=mu+eps*torch.exp(logvar/2)
        return mu
        #return z
    

    
    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        
        x, _ = batch
        x = x.view(x.size(0), -1)   
        out1=self.encoder(x)
        mu=self.get_mu(out1.view(out1.size(0),-1))#64,128*7*7->64,32
        out2=self.encoder(x)
        logvar=self.get_logvar(out2.view(out2.size(0),-1))

        z=self.get_z(mu,logvar)
        out3=self.get_temp(z).view(z.size(0),128,32,32)
        #out3=self.get_temp(z).view(z.size(0),128,7,7)
        #generated_imgs = z.view(z.size(0), 1, 256, 192)
        out = self.decoder(out3)
        loss, bce, kld = loss_function(out, x, mu, log_var)
        self.log("train_loss", loss)
        return {"loss": loss, "pred": pred}
    
    def training_step_end(self, batch_parts):
        # predictions from each GPU
        predictions = batch_parts["pred"]
        # losses from each GPU
        losses = batch_parts["loss"]

        gpu_0_prediction = predictions[0]
        gpu_1_prediction = predictions[1]

        # do something with both outputs
        return (losses[0] + losses[1]) / 2
    
    def validation_step(self, batch, batch_idx):
        
        x, _ = batch
        x = x.view(x.size(0), -1)   
        out1=self.encoder(x)
        mu=self.get_mu(out1.view(out1.size(0),-1))#64,128*7*7->64,32
        out2=self.encoder(x)
        logvar=self.get_logvar(out2.view(out2.size(0),-1))

        z=self.get_z(mu,logvar)
        out3=self.get_temp(z).view(z.size(0),128,32,32)
        #out3=self.get_temp(z).view(z.size(0),128,7,7)
        #generated_imgs = z.view(z.size(0), 1, 256, 192)
        out = self.decoder(out3)
        loss, bce, kld = loss_function(out, x, mu, log_var)
        self.log("val_loss", loss)
        return {"loss": loss, "pred": pred}
    
    def validation_step_end(self, batch_parts):
        # predictions from each GPU
        predictions = batch_parts["pred"]
        # losses from each GPU
        losses = batch_parts["loss"]

        gpu_0_prediction = predictions[0]
        gpu_1_prediction = predictions[1]

        # do something with both outputs
        return (losses[0] + losses[1]) / 2
    
    def loss_function(recon_x, x, mu, log_var):
        L1_loss = nn.L1Loss(reduction='sum')#
        BCE = L1_loss(recon_x, x)
        L2_loss = nn.MSELoss()
        MSE = L2_loss (recon_x, x)
        #BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        #return BCE + KLD, BCE, KLD
        #loss = BCE+ 0.000000000002 * KLD
        loss = BCE
        return loss
    
# =============================================================================
#     def training_step(self, batch, batch_idx):
#         # training_step defined the train loop.
#         # It is independent of forward
#         x, y = batch
#         x = x.view(x.size(0), -1)
#         z = self.encoder(x)
#         x_hat = self.decoder(z)
#         loss = F.mse_loss(x_hat, x)
#         # Logging to TensorBoard by default
#         self.log("train_loss", loss)
#         return loss
# =============================================================================

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        num_training_samples = len(self.trainer.datamodule.train_dataloader())
        return optimizer
    
    def configure_callbacks(self):
        early_stop = EarlyStopping(monitor="val_acc", mode="max")
        checkpoint = ModelCheckpoint(monitor="val_loss")
        return [early_stop, checkpoint]
