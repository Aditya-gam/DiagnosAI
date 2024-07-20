import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Load the dataset
data = pd.read_csv('./input/NIH_Dataset/Data_Entry_2017.csv')
data = data[data['Patient Age'] < 100]  # Filter out entries with age >= 100

# Map image paths
base_path = './input/NIH_Dataset/images'
data['path'] = data['Image Index'].apply(lambda x: os.path.join(base_path, x))

# Remove 'No Finding' labels and split labels
data['Finding Labels'] = data['Finding Labels'].apply(lambda x: x.replace('No Finding', ''))
all_labels = sorted(np.unique(np.concatenate(data['Finding Labels'].map(lambda x: x.split('|')).values)))

# Create binary labels
for label in all_labels:
    data[label] = data['Finding Labels'].apply(lambda x: 1.0 if label in x else 0)

# Define transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize to 128x128
    transforms.Grayscale(),  # Convert to grayscale
    transforms.RandomHorizontalFlip(),  # Random horizontal flip
    transforms.RandomRotation(5),  # Random rotation by 5 degrees
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1] range
])

class ChestXrayDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['path']
        image = Image.open(img_path).convert('RGB')  # Open the image file
        label = self.dataframe.iloc[idx][all_labels].values.astype(np.float32)

        if self.transform:
            image = self.transform(image)

        return image, label

# Create dataset
dataset = ChestXrayDataset(data, transform=transform)

# Splitting the dataset
train_size = int(0.75 * len(dataset))
valid_size = len(dataset) - train_size
train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=False)

# Function to visualize images
def show_images(images, labels, n=4):
    fig, axs = plt.subplots(n, n, figsize=(10, 10))
    axs = axs.flatten()
    for img, label, ax in zip(images, labels, axs):
        img = img.numpy().transpose((1, 2, 0))
        img = np.clip(img, 0, 1)
        ax.imshow(img[:, :, 0], cmap='gray')
        ax.set_title(', '.join([all_labels[i] for i in range(len(label)) if label[i] > 0.5]))
        ax.axis('off')
    plt.show()

# Visualize some training images
images, labels = next(iter(train_loader))
show_images(images, labels)