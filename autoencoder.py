import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.optim import Adam
import numpy as np

class Autoencoder(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        kernel_size = 4
        self.conv1 = nn.Conv2d(cfg['nc'], cfg['nfe'], kernel_size, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cfg['nfe'])
        self.conv2 = nn.Conv2d(cfg['nfe'], cfg['nfe'] * 2, kernel_size, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cfg['nfe'] * 2)
        self.conv3 = nn.Conv2d(cfg['nfe'] * 2, cfg['nz'], kernel_size, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(cfg['nz'])

        self.deconv1 = nn.ConvTranspose2d(cfg['nz'], cfg['nfd'] * 2, kernel_size, 1, 0, bias=False)
        self.bn1_ = nn.BatchNorm2d(cfg['nfd'] * 2)
        self.deconv2 = nn.ConvTranspose2d(cfg['nfd'] * 2, cfg['nfd'], kernel_size, 2, 1, bias=False)
        self.bn2_ = nn.BatchNorm2d(cfg['nfd'])
        self.deconv3 = nn.ConvTranspose2d(cfg['nfd'], cfg['nc'], kernel_size, 2, 1, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        z = self.encoder(x)
        x_ = self.decoder(z)
        return x_

    def encoder(self,x):
        #print(x.shape)
        x = F.relu(self.bn1(self.conv1(x)))
        #print(x.shape)
        x = F.relu(self.bn2(self.conv2(x)))
        #print(x.shape)
        z = F.relu(self.bn3(self.conv3(x)))
        #print(z.shape)
        return z

    def decoder(self,z):
        x = F.relu(self.bn1_(self.deconv1(z)))
        #print(x.shape)
        x = F.relu(self.bn2_(self.deconv2(x)))
        #print(x.shape)
        x_ = self.sig(self.deconv3(x))
        #print(x_.shape)
        return x_

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.cfg['lr'])

    def training_step(self, batch, batch_idx):
        x, _ = batch
        output = self(x)
        loss = F.mse_loss(output, x)
        self.log('loss', loss, on_step=True, prog_bar=False)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        output = self(x)
        loss = F.mse_loss(output, x)

        # save input and output images at beginning of epoch
        if batch_idx == 0:
            self.save_images(x, output, "val_input_output")

        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log('avg_val_loss', avg_loss, on_epoch=True, prog_bar=True)

    def save_images(self, x, output, name, n=8):
        """
        Saves a plot of n images from input and output batch
        """
        # make grids and save to logger
        grid_top = vutils.make_grid(x[:n,:1,:,:], nrow=n)
        grid_bottom = vutils.make_grid(output[:n,:1,:,:], nrow=n)
        grid = torch.cat((grid_top, grid_bottom), 1)
        self.logger.experiment.add_image(name, grid, self.current_epoch)
