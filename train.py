from argparse import ArgumentParser
import os
import sys
import cv2
import numpy as np

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer, loggers
from torchsummary import summary
import torch.nn.functional as F


def train(cfg):
    pl.seed_everything(42)
    logger = pl.loggers.TensorBoardLogger(os.path.join('lightning_logs/',cfg['experiment']))

    from pump_datamodule import PumpDataModule
    dm = PumpDataModule(cfg)
    dm.setup(stage='fit')

    from autoencoder import Autoencoder
    model = Autoencoder(cfg)
    trainer = Trainer(gpus=cfg['gpus'], max_epochs=cfg['max_epoch'], logger=logger)
    trainer.fit(model, dm)

    model_dir = os.path.join("trained_models/",cfg['experiment'])
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(model, os.path.join(model_dir,"model.pt"))


if __name__ == "__main__":


    cfg = {
           'experiment': 'simpel',
           'train_folder': 'data/train',
           'val_folder': 'data/val/',
           'dim': (5,20,20),
           'max_epoch': 250,
           'gpus': 1,
           'lr': 0.0005,
           'batch_size': 32,
           'nc': 5,
           'nz': 8,
           'nfe': 32,
           'nfd': 32,
           'device': 'cuda',
          }

    train(cfg)
