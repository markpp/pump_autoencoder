import os
import sys
import cv2
import numpy as np
from glob import glob

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer, loggers
from torchsummary import summary
import torch.nn.functional as F



def test(cfg, dataset='val', show=False, save=True):

    from pump_dataset import PumpDataset

    if dataset == 'train':
        data = PumpDataset(data_dir=cfg['train_folder'])
    elif dataset == 'val':
        data = PumpDataset(data_dir=cfg['val_folder'])
    elif dataset == 'test':
        data = PumpDataset(data_dir=cfg['test_folder'])

    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    from autoencoder import Autoencoder
    model = Autoencoder(cfg)

    model_dir = os.path.join("trained_models/",cfg['experiment'])
    model = torch.load(os.path.join(model_dir,"model.pt"))
    model.eval()
    with torch.no_grad():
        print("number of samples {}".format(len(data)))
        inputs, recs, files, losses, latent = [], [], [], [], []
        for i, data in enumerate(data):
            sample, path = data
            z = model.encoder(sample.unsqueeze(0))
            rec = model.decoder(z)[0]
            loss = F.mse_loss(rec, sample)
            inputs.append(sample.numpy())
            files.append(path)
            latent.append(z[0].numpy().flatten())
            rec = rec.numpy()
            recs.append(rec)
            losses.append(loss.item())

        if save:
            np.save(os.path.join(output_dir,'{}_inputs.npy'.format(dataset)), inputs)
            np.save(os.path.join(output_dir,'{}_recs.npy'.format(dataset)), recs)
            np.save(os.path.join(output_dir,'{}_files.npy'.format(dataset)), files)
            np.save(os.path.join(output_dir,'{}_losses.npy'.format(dataset)), losses)
            np.save(os.path.join(output_dir,'{}_latent.npy'.format(dataset)), latent)

if __name__ == "__main__":
    cfg = {
           'experiment': 'simpel',
           'train_folder': 'data/train',
           'val_folder': 'data/val/',
           'dim': (5,20,20),
           'max_epoch': 100,
           'gpus': 1,
           'lr': 0.0005,
           'batch_size': 32,
           'nc': 5,
           'nz': 8,
           'nfe': 32,
           'nfd': 32,
           'device': 'cuda',
          }

    test(cfg,dataset='val')
    test(cfg,dataset='train')
