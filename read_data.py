# encoding: utf-8

"""
Read images and corresponding labels.
"""
import pickle
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from PIL import Image, ImageFile
import pandas as pd
import os
ImageFile.LOAD_TRUNCATED_IMAGES = True

random.seed(0)
    
def augument(image_batch, transform, patient2image, image2patient):
    aug1, aug2 = [], []
    for image in image_batch:
        patient_id = image2patient[image]
        patient_images = patient2image[patient_id]
        num1, num2 = random.randint(0, len(patient_images) - 1), random.randint(0, len(patient_images) - 1)
        image = Image.open(patient_images[num1]).convert('RGB')
        aug1.append(transform(image))
        image = Image.open(patient_images[num2]).convert('RGB')
        aug2.append(transform(image))
    
    return torch.stack(aug1), torch.stack(aug2)
        

class ChestXrayDataSet(Dataset):
    def __init__(self, data_dir, image_list_file, transform=None):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        image_names = []
        labels = []
        with open(image_list_file, "r") as f:
            i = 0
            for line in f:
                items = line.split()
                image_name= items[0]
                label = items[1:]
                label = [int(i) for i in label]
                label = label[:8]
                image_name = os.path.join(data_dir, image_name)
                image_names.append(image_name)
                labels.append(label)
#                 i += 1
#                 if i == 22051:
#                     break

        self.image_names = image_names
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item

        Returns:
            image and its labels
        """
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image_name, image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names)