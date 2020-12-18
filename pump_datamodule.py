import torch
import torchvision
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl

import os
from glob import glob
import cv2

from pump_dataset import PumpDataset

class PumpDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage='fit'):
        if stage == 'fit':
            self.data_train = PumpDataset(data_dir=self.cfg['train_folder'])
            self.data_val = PumpDataset(data_dir=self.cfg['val_folder'])

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.cfg['batch_size'], shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.cfg['batch_size'], shuffle=False)


if __name__ == '__main__':

    cfg = {
           'experiment': 'simpel',
           'train_folder': 'data/train',
           'val_folder': 'data/val/',
           'dim': (4,16,16),
           'max_epoch': 200,
           'gpus': 1,
           'lr': 0.0005,
           'batch_size': 16,
           'nc': 5,
           'nz': 8,
           'nfe': 32,
           'nfd': 32,
           'device': 'cuda',
          }

    dm = PumpDataModule(cfg)
    dm.setup()

    # cleanup output dir
    import os, shutil
    output_root = "output/"
    if os.path.exists(output_root):
        shutil.rmtree(output_root)
    os.makedirs(output_root)

    sample_idx = 0
    for batch_id, batch in enumerate(dm.val_dataloader()):
        samples, paths = batch
        print(samples.shape)
