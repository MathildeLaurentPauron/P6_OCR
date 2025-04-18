import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np


class DogsDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df  # ✅ Stocker tout le DataFrame
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]["filename"]  # ✅ Accéder au chemin depuis df
        label = self.df.iloc[idx]["breed"]  # ✅ Récupérer le label

        img = Image.open(img_path)
        if self.transforms:
            # for transform in self.transforms:
            img = self.transforms(img)

        return img, label

preprocessing = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        #transforms.GaussianBlur(3),
        transforms.ToTensor(),
    ]
)
data_augment = transforms.Compose(
    [
      transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
      transforms.RandomHorizontalFlip(0.5),
      transforms.RandomRotation(10),
      transforms.ToTensor(),
    ]
)

preprocessing_transfert = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        #transforms.GaussianBlur(3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalisation ImageNet
    ]
)
data_augment_transfert = transforms.Compose(
    [
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalisation ImageNet
    ]
)