import os
import numpy as np
import torch
import torch.utils.data
import argparse

import cv2
import random
import math
from glob import glob
import pickle


class PumpDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.data_list = sorted([y for x in os.walk(data_dir) for y in glob(os.path.join(x[0], '*.pkl'))])
        if not len(self.data_list)>0:
            print("did not find any files")

    def load_sample(self, path):
        sample = pickle.load(open(path, 'rb'))
        return sample, path

    def __getitem__(self, idx):
        sample, path = self.load_sample(self.data_list[idx])

        sample = torch.from_numpy(sample[:20,:20,:])
        sample = sample.permute(2, 0, 1) # (HxHxP -> PxHxH)
        sample = sample.float()
        return sample, path

    def __len__(self):
        return len(self.data_list)


if __name__ == '__main__':
    dataset = PumpDataset(data_dir='data/val/')

    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    import matplotlib.pyplot as plt

    print(len(dataset))
    for i, data in enumerate(dataset):
        sample, path = data
        sample = sample.numpy()

        n_pump, n_freq, n_time = sample.shape

        fig, ax = plt.subplots(1,n_pump,figsize=(4*n_pump,4))

        # Plot for hver pumpe
        for pump in range(n_pump):
            ax[pump].imshow(sample[pump,:,:])
            ax[pump].set_title('Pump ' + str(pump))
            ax[pump].set_xlabel('Hour')
            ax[pump].set_ylabel('Period')

        fig.tight_layout()
        plt.savefig(os.path.join(output_dir,'{}.png'.format(i)))
        break
