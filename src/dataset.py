import os
import pandas as pd
import numpy as np
from PIL import Image
from glob import glob
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from itertools import chain
import config


def load_image_lists():
    """
    Load lists of image filenames for the training/validation and test datasets.

    This function reads two separate text files: 'train_val_list.txt' and 'test_list.txt',
    which are expected to contain one image file name per line. These files are located
    in a directory specified by `config.BASE_DATA_PATH`. The function splits the content
    of each file into a list of filenames based on newline characters.

    Returns:
        tuple: A tuple containing two lists:
               - The first list contains the filenames for the training and validation images.
               - The second list contains the filenames for the test images.
    """
    # Read the list of training and validation image filenames.
    # Uses a context manager to ensure the file is properly closed after reading.
    with open(os.path.join(config.BASE_DATA_PATH, 'train_val_list.txt'), 'r') as file:
        # Read and split the file content into a list of strings.
        train_val_images = file.read().splitlines()

    # Read the list of test image filenames.
    with open(os.path.join(config.BASE_DATA_PATH, 'test_list.txt'), 'r') as file:
        # Read and split the file content into a list of strings.
        test_images = file.read().splitlines()

    # Return the lists of filenames as a tuple.
    return train_val_images, test_images


def load_data():
    """
    Load and preprocess the dataset for training and testing.

    This function performs several key operations:
    1. Loads the lists of training/validation and testing image filenames.
    2. Reads the dataset metadata from a CSV file and filters out records with patient age >= 100.
    3. Constructs full paths to image files and associates them with the dataset records.
    4. Addresses class imbalance by resampling the dataset based on the prevalence of findings.
    5. Splits the data into training/validation and testing subsets based on the image lists.
    6. Processes labels to handle cases with 'No Finding' and converts labels to binary format for all classes.
    7. Filters labels to include only those with at least 1000 occurrences in the training/validation dataset.

    Returns:
        tuple: Contains three elements:
               - train_val_data: DataFrame with training/validation data.
               - test_data: DataFrame with testing data.
               - valid_labels: List of labels that have at least 1000 occurrences.
    """
    # Load lists of image files for training/validation and testing.
    train_val_images, test_images = load_image_lists()

    # Load dataset metadata from the configured CSV file and filter out entries with patient age >= 100.
    data = pd.read_csv(config.CSV_FILE)
    data = data[data['Patient Age'] < 100]

    # Construct the full path for each image file in the dataset.
    data['image_file'] = data['Image Index'].apply(
        lambda x: x.split('.')[0] + '.png')

    # Use glob to match the image file pattern and create a dictionary mapping file names to their full paths.
    image_paths = glob(config.IMAGE_PATH_PATTERN)
    image_path_dict = {os.path.basename(path): path for path in image_paths}
    data['path'] = data['image_file'].map(image_path_dict.get)

    # Address class imbalance by calculating sample weights and resampling the data.
    sample_weights = data[config.FINDING_LABELS_COLUMN].map(
        lambda x: len(x.split('|')) if x else 0).values + 0.04
    sample_weights /= sample_weights.sum()
    # Random state for reproducibility.
    data = data.sample(40000, weights=sample_weights, random_state=42)

    # Split the data into training/validation and testing sets.
    train_val_data = data[data['image_file'].isin(train_val_images)].copy()
    test_data = data[data['image_file'].isin(test_images)].copy()

    # Process the 'Finding Labels' column to remove 'No Finding' and split labels.
    train_val_data.loc[:, config.FINDING_LABELS_COLUMN] = train_val_data[config.FINDING_LABELS_COLUMN].apply(
        lambda x: x.replace('No Finding', ''))
    test_data.loc[:, config.FINDING_LABELS_COLUMN] = test_data[config.FINDING_LABELS_COLUMN].apply(
        lambda x: x.replace('No Finding', ''))

    # Convert labels from string to a set of binary columns, one for each label.
    all_labels = sorted(set(chain(
        *train_val_data[config.FINDING_LABELS_COLUMN].map(lambda x: filter(None, x.split('|'))))))
    for label in all_labels:
        train_val_data[label] = train_val_data[config.FINDING_LABELS_COLUMN].apply(
            lambda x: 1 if label in x else 0)
        test_data[label] = test_data[config.FINDING_LABELS_COLUMN].apply(
            lambda x: 1 if label in x else 0)

    # Filter labels to keep only those with at least 1000 occurrences in the training/validation dataset.
    label_counts = train_val_data[all_labels].sum()
    valid_labels = label_counts[label_counts >= 1000].index.tolist()

    # Reduce the data to include only paths, image files, and valid labels.
    train_val_data = train_val_data.loc[:, [
        'path', 'image_file'] + valid_labels]
    test_data = test_data.loc[:, ['path', 'image_file'] + valid_labels]

    return train_val_data, test_data, valid_labels


def get_transforms():
    """
    Define and return a composition of image transformations.

    This function sets up a series of transformations to preprocess images for model input. 
    These transformations include resizing images to a fixed dimension, applying random horizontal flips and
    rotations for data augmentation, converting images to tensor format, and normalizing them with pre-defined mean 
    and standard deviation values. These transformations are commonly used to standardize inputs and enhance the
    diversity of the training data, which can help improve model generalization.

    Returns:
        torchvision.transforms.Compose: A composed series of transformations.
    """
    # Define a composition of image transformations for preprocessing.
    return transforms.Compose([
        transforms.Resize((224, 224)),  # Resize all images to 224x224 pixels.
        # Randomly flip images horizontally to augment the dataset.
        transforms.RandomHorizontalFlip(),
        # Randomly rotate images by up to 5 degrees to introduce variability.
        transforms.RandomRotation(5),
        # Convert images to PyTorch tensors (scale from 0 to 1).
        transforms.ToTensor(),
        # Normalize tensor images using predefined mean and std.
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])


class ChestXrayDataset(Dataset):
    """
    A custom dataset class for chest X-ray images, tailored for use with PyTorch models.

    This dataset class inherits from PyTorch's Dataset and is intended to facilitate the
    loading and preprocessing of image data for model training or evaluation. The class
    expects a pandas DataFrame containing image file paths and labels, along with optional
    transformations.

    Attributes:
        dataframe (pd.DataFrame): DataFrame containing the image paths and associated metadata.
        labels (list): List of strings representing the columns in the DataFrame that contain the labels.
        transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version.

    Methods:
        __len__: Returns the total number of samples in the dataset.
        __getitem__: Retrieves a sample and its corresponding label at a given index.
    """

    def __init__(self, dataframe, labels, transform=None):
        """
        Initialize the dataset with the DataFrame, labels, and optional transforms.

        Parameters:
            dataframe (pd.DataFrame): The DataFrame containing the image paths and their corresponding labels.
            labels (list): List of columns in the DataFrame that contain the labels for the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dataframe = dataframe  # Store the input dataframe containing paths and labels.
        # Store the list of label column names from the dataframe.
        self.labels = labels
        # Optional transform (augmentation) to apply to each image.
        self.transform = transform

    def __len__(self):
        """
        Return the number of items in the dataset.

        Returns:
            int: The total number of samples in the dataset.
        """
        return len(self.dataframe)

    def __getitem__(self, idx):
        """
        Retrieve the image and labels at the specified index.

        Parameters:
            idx (int): The index of the item.

        Returns:
            tuple: Contains the image and its corresponding labels. The image is returned as a PIL image,
                   potentially transformed if a transform was provided. The labels are returned as a numpy array.
        """
        # Get the image path from the dataframe at the specified index.
        img_path = self.dataframe.iloc[idx]['path']
        # Open the image file at the path, convert it to RGB to ensure three channels.
        image = Image.open(img_path).convert('RGB')

        # Retrieve the labels from the dataframe using the provided label columns, converted to a numpy float32 array.
        label = self.dataframe.iloc[idx][self.labels].values.astype(np.float32)

        # If a transform is specified, apply it to the image.
        if self.transform:
            image = self.transform(image)

        # Return the image and its labels as a tuple.
        return image, label


def prepare_data_loaders(data, all_labels, batch_size=config.BATCH_SIZE, val_batch_size=config.VAL_BATCH_SIZE):
    """
    Prepare DataLoader objects for training and validation.

    This function organizes the data into stratified subsets if possible and creates DataLoader objects
    for both training and validation. Stratification is based on the presence of labels with at least 2 samples,
    ensuring that both training and validation sets are representative of the overall dataset.

    Parameters:
        data (pd.DataFrame): The DataFrame containing all image data and labels.
        all_labels (list): A list of all possible labels.
        batch_size (int): The batch size for the training DataLoader.
        val_batch_size (int): The batch size for the validation DataLoader.

    Returns:
        tuple: Contains two DataLoader objects for training and validation datasets.
    """
    # Print initial data shape and label distribution for debugging purposes.
    print("Data shape before splitting:", data.shape)
    label_sums = data[all_labels].sum()
    print("Label sums:\n", label_sums)

    # Identify which labels have enough samples for stratification (at least 2 samples).
    stratifiable_labels = label_sums[label_sums >= 2].index.tolist()
    print("Stratifiable labels:", stratifiable_labels)

    # Handling scenarios with insufficient samples for any meaningful stratification.
    if not stratifiable_labels:
        print("Warning: No labels have at least 2 samples; stratification will not be performed.")
        train_data, valid_data = train_test_split(
            data, test_size=0.25, random_state=42)
    else:
        # Find the label with the maximum samples to use as the primary key for stratification.
        primary_label = label_sums.idxmax()
        print("Primary label for stratification:", primary_label)

        # Perform stratified split if possible; otherwise, fall back to a simple split.
        if data[primary_label].nunique() >= 2:
            train_data, valid_data = train_test_split(
                data, test_size=0.25, stratify=data[primary_label], random_state=42)
        else:
            print(
                "Insufficient classes for stratification on primary label. Performing a simple split.")
            train_data, valid_data = train_test_split(
                data, test_size=0.25, random_state=42)

    print("Train data shape:", train_data.shape)
    print("Validation data shape:", valid_data.shape)

    # Create dataset objects for training and validation using the specified transformations.
    train_dataset = ChestXrayDataset(
        train_data, all_labels, transform=get_transforms())
    valid_dataset = ChestXrayDataset(
        valid_data, all_labels, transform=get_transforms())

    # Create DataLoader objects to manage batch loading of data, with shuffling for training set.
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(
        valid_dataset, batch_size=val_batch_size, shuffle=False)

    return train_loader, valid_loader


def prepare_test_loader(test_data, all_labels, batch_size=config.VAL_BATCH_SIZE):
    """
    Prepare a DataLoader for the test dataset.

    This function creates a DataLoader object for the test dataset, ensuring the data is processed 
    with the same transformations as the training and validation data. It's designed to facilitate the
    evaluation of a model on the test set, providing batched, preprocessed data for prediction.

    Parameters:
        test_data (pd.DataFrame): DataFrame containing the test data with image paths and labels.
        all_labels (list): List of label names corresponding to the columns in the test_data DataFrame.
        batch_size (int): The number of samples to load per batch. Uses the validation batch size by default.

    Returns:
        DataLoader: Configured DataLoader for the test data set with specified batch size and no shuffling.
    """
    # Create a dataset object for the test data. Apply the same transformations as for the training/validation datasets.
    test_dataset = ChestXrayDataset(
        test_data, all_labels, transform=get_transforms())

    # Create a DataLoader object for the test dataset.
    # Set shuffle to False because shuffling is not needed during testing/evaluation.
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader
