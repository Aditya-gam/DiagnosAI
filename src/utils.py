import os
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from model import ResNet50Model


def initialize_model(device, num_classes, feature_extract=True, optimizer_name='SGD', lr=0.0001, momentum=0.9, weight_decay=0.0, step_size=7, gamma=0.1):
    """
    Initialize the ResNet50 model with specified configurations, including the optimizer and learning rate scheduler.

    This function sets up a ResNet50 model for a classification task, configures its optimizer and learning rate
    scheduler based on the provided arguments. It also sets up the loss function suitable for multi-label classification.

    Parameters:
        device (torch.device): The device to train the model on (e.g., 'cuda' or 'cpu').
        num_classes (int): Number of classes for the output layer.
        feature_extract (bool, optional): Whether to use feature extraction. If True, model weights are frozen except the final layer.
        optimizer_name (str, optional): Choice of optimizer ('SGD', 'Adam', or 'AdamW').
        lr (float, optional): Learning rate for the optimizer.
        momentum (float, optional): Momentum factor (relevant only for SGD).
        weight_decay (float, optional): Weight decay (L2 penalty) for regularization.
        step_size (int, optional): Period of learning rate decay.
        gamma (float, optional): Multiplicative factor of learning rate decay.

    Returns:
        tuple: Contains the initialized model, loss criterion, optimizer, and learning rate scheduler.
    """
    # Initialize the model with the specified number of classes and feature extraction setting.
    model = ResNet50Model(num_classes, feature_extract, use_attention=False)
    model = model.to(device)  # Move model to the specified device.

    # Define the criterion for multi-label classification.
    criterion = torch.nn.BCEWithLogitsLoss()

    # Initialize the optimizer based on the specified type and parameters.
    if optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr,
                              momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr,
                               weight_decay=weight_decay)
    elif optimizer_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr,
                                weight_decay=weight_decay)
    else:
        # Handle the case of an unsupported optimizer.
        raise ValueError("Unsupported optimizer")

    # Set up the learning rate scheduler to adjust the learning rate based on epochs.
    scheduler = lr_scheduler.StepLR(
        optimizer, step_size=step_size, gamma=gamma)

    return model, criterion, optimizer, scheduler


def save_model(model, path):
    """
    Saves the model's state dictionary to a specified file path.

    This function ensures the directory for the path exists before saving the model. If the directory does not
    exist, it creates the necessary directories. This is useful for saving models during or after training,
    allowing for model persistence across sessions.

    Parameters:
        model (torch.nn.Module): The PyTorch model whose state_dict is to be saved.
        path (str): The file path where the model state dictionary should be saved. This includes the directory
                    and the file name (e.g., './models/model.pth').

    Examples:
        save_model(model, './models/my_model.pth')  # Saves the model's state_dict to 'my_model.pth' under 'models' directory.
    """
    # Ensure the directory exists where the model will be saved.
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)  # Create the directory if it does not exist.

    # Save the model's state dictionary to the specified path.
    torch.save(model.state_dict(), path)


def create_metrics_list(metrics_history, train_loss, train_acc, val_loss, val_acc, f1, precision, recall, auc):
    """
    Update and return the metrics history dictionary with the latest metrics from a training cycle.

    This function appends the latest training and validation metrics to their respective lists in the
    metrics_history dictionary. It is useful for tracking the progression of model performance over multiple
    epochs, allowing for later analysis and visualization of trends such as loss reduction and accuracy improvements.

    Parameters:
        metrics_history (dict): Dictionary containing lists of metrics collected over epochs.
                                The dictionary should have the keys: 'train_loss', 'train_acc', 'val_loss',
                                'val_acc', 'f1', 'precision', 'recall', and 'auc'.
        train_loss (float): Training loss for the current epoch.
        train_acc (float): Training accuracy for the current epoch.
        val_loss (float): Validation loss for the current epoch.
        val_acc (float): Validation accuracy for the current epoch.
        f1 (float): F1 score for the validation set in the current epoch.
        precision (float): Precision score for the validation set in the current epoch.
        recall (float): Recall score for the validation set in the current epoch.
        auc (float): AUC score for the validation set in the current epoch.

    Returns:
        dict: The updated metrics_history dictionary containing the appended values.

    Example:
        # Example dictionary for initialization
        metrics_history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'f1': [], 'precision': [], 'recall': [], 'auc': []
        }

        # Example call to the function
        updated_metrics = create_metrics_list(metrics_history, 0.5, 0.8, 0.4, 0.85, 0.75, 0.8, 0.77, 0.85)
    """
    # Append the current epoch's metrics to their respective lists in the dictionary.
    metrics_history['train_loss'].append(train_loss)
    metrics_history['train_acc'].append(train_acc)
    metrics_history['val_loss'].append(val_loss)
    metrics_history['val_acc'].append(val_acc)
    metrics_history['f1'].append(f1)
    metrics_history['precision'].append(precision)
    metrics_history['recall'].append(recall)
    metrics_history['auc'].append(auc)

    return metrics_history


def count_parameters(model):
    """
    Count the total, trainable, and non-trainable parameters in a PyTorch model.

    This function computes the number of total parameters in the model, differentiates between
    trainable and non-trainable parameters, and returns these counts. Trainable parameters are those
    which will be updated during training (i.e., where requires_grad is True), while non-trainable
    parameters remain static during training.

    Parameters:
        model (torch.nn.Module): The PyTorch model for which parameters are counted.

    Returns:
        tuple: A tuple containing three elements:
            - total_params (int): The total number of parameters in the model.
            - trainable_params (int): The number of trainable parameters.
            - non_trainable_params (int): The number of non-trainable parameters.

    Example:
        model = ResNet50Model(num_classes=10)
        total_params, trainable_params, non_trainable_params = count_parameters(model)
        print(f"Total: {total_params}, Trainable: {trainable_params}, Non-Trainable: {non_trainable_params}")
    """
    # Calculate the total number of parameters.
    total_params = sum(p.numel() for p in model.parameters())
    # Calculate the number of trainable parameters.
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    # Calculate the number of non-trainable parameters.
    non_trainable_params = total_params - trainable_params

    return total_params, trainable_params, non_trainable_params
