from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import utils
from datasets import FaceLandmarksDataset, Rescale, RandomCrop, ToTensor

if __name__ == '__main__':
    face_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv',
                                        root_dir='data/faces/')

    fig = plt.figure()

    for i in range(len(face_dataset)):
        sample = face_dataset[i]

        print(i, sample['image'].shape, sample['landmarks'].shape)

        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        utils.show_landmarks(**sample)  # ** desempaquetar diccionarios

        if i == 3:
            plt.show()
            break

    scale = Rescale(256)
    crop = RandomCrop(128)
    composed = transforms.Compose([Rescale(256),
                                   RandomCrop(224)])

    # Apply each of the above transforms on sample.
    fig = plt.figure()
    sample = face_dataset[65]
    for i, tsfrm in enumerate([scale, crop, composed]):
        transformed_sample = tsfrm(sample)

        ax = plt.subplot(1, 3, i + 1)
        plt.tight_layout()
        ax.set_title(type(tsfrm).__name__)
        utils.show_landmarks(**transformed_sample)

    plt.show()

    # Crear el dataset con transformaciones:
    transformed_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv',
                                               root_dir='data/faces/',
                                               transform=transforms.Compose([
                                                   Rescale(256),
                                                   RandomCrop(224),
                                                   ToTensor()
                                               ]))

    # esto se cambia por un Dataloader:
    """for i in range(len(transformed_dataset)):
        sample = transformed_dataset[i]

        print(i, sample['image'].size(), sample['landmarks'].size())

        if i == 3:
            break"""

    dataloader = DataLoader(transformed_dataset, batch_size=4,
                            shuffle=True, num_workers=0)

    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(),
              sample_batched['landmarks'].size())

        # observe 4th batch and stop.
        if i_batch == 3:
            plt.figure()
            utils.show_landmarks_batch(sample_batched)
            plt.axis('off')
            plt.ioff()
            plt.show()
            break