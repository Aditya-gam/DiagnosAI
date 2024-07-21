# dataset.py
import os
import pandas as pd
import numpy as np
from PIL import Image
from glob import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from itertools import chain
import config

def load_data():
    data = pd.read_csv(config.CSV_FILE)
    data = data[data['Patient Age'] < 100]
    image_paths = glob(config.IMAGE_PATH_PATTERN)
    image_path_dict = {os.path.basename(path): path for path in image_paths}
    data['path'] = data['Image Index'].map(image_path_dict.get)
    data[config.FINDING_LABELS_COLUMN] = data[config.FINDING_LABELS_COLUMN].apply(lambda x: x.replace('No Finding', ''))
    all_labels = sorted(np.unique(list(chain(*data[config.FINDING_LABELS_COLUMN].map(lambda x: x.split('|')).tolist()))))
    for label in all_labels:
        data[label] = data[config.FINDING_LABELS_COLUMN].apply(lambda x, lbl=label: 1.0 if lbl in x else 0)
    return data, all_labels

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

class ChestXrayDataset(Dataset):
    def __init__(self, dataframe, labels, transform=None):
        self.dataframe = dataframe
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['path']
        image = Image.open(img_path).convert('RGB')
        label = self.dataframe.iloc[idx][self.labels].values.astype(np.float32)
        if self.transform:
            image = self.transform(image)
        return image, label

def prepare_data_loaders(data, all_labels, batch_size=config.BATCH_SIZE, val_batch_size=config.VAL_BATCH_SIZE):
    dataset = ChestXrayDataset(data, all_labels, transform=get_transforms())
    train_size = int(0.75 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=val_batch_size, shuffle=False)

    return train_loader, valid_loader
