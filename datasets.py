from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import json
import cv2


############ FACE LANDMARKS DATASET ##########################
class FaceLandmarksDataset(Dataset):
    """dataset"""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample


############ END FACE LANDMARKS DATASET ##########################


############ FACE LANDMARKS DATASET TRANSFORMS ##########################
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                left: left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}


############ END FACE LANDMARKS DATASET TRANSFORMS ##########################

########### CITYSCAPES DATASET ###############################
class CityScapesDataset(Dataset):

    def __init__(self, json_dir, img_dir, classes, transform=None):
        """
        Args:
            json_dir (string): Directory to the annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        txt_json = open(json_dir)
        self.jsons = txt_json.readlines()
        txt_img = open(img_dir)
        self.images = txt_img.readlines()
        #self.jsons = list([f for f in sorted(os.listdir(json_dir)) if f.endswith('.json')])
        #self.images = list(sorted(os.listdir(img_dir)))
        self.transform = transform
        self.classes = classes

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        from PIL import Image
        img = io.imread(os.path.join(self.img_dir, self.images[idx]))/255.

        with open(os.path.join(self.json_dir, self.jsons[idx])) as f:
            data = json.load(f)

        h = data['imgHeight']
        w = data['imgWidth']

        # creacion de las mascaras de los objetos indicados
        masks = np.zeros((1, h, w), dtype=np.uint8)
        mask = np.zeros((h, w), dtype=np.uint8)
        boxes = []
        labels = []
        for object in data['objects']:
            if object['label'] in self.classes:
                idx = self.classes.index(object['label'])
                points = np.array(object['polygon'], np.int32)
                mask = cv2.fillConvexPoly(mask, points, 255) / 255.0
                masks = np.concatenate((masks, np.expand_dims(mask, axis=0)), axis=0)

                # cálculo de los bbx de los objetos
                points = object["polygon"]
                points = np.array(points, np.int32)
                tag = self.classes.index(object["label"])
                x, y, w, h = cv2.boundingRect(np.array(points))
                x_min = x
                x_max = x + w
                y_min = y
                y_max = y + h
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(tag)

        masks = masks[1:, :, :]
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transform is not None:
            img, target = self.transform(img, target)
        img = np.transpose(img, (2, 0, 1))

        return img, target


##################CITYSCAPES DATALOADER ##########################
def dataLoaderCitysCapes(batch):
    return tuple(zip(*batch))
