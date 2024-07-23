import torch
import numpy as np
from utils import save_model, create_metrics_list
from evaluate import calculate_metrics


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch over the entire training dataset.

    This function iteratively processes batches of data, computes the loss, performs backpropagation,
    and updates the model's weights. Additionally, it calculates and returns the average loss and accuracy
    for the epoch.

    Parameters:
        model (torch.nn.Module): The model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        criterion (torch.nn.Module): Loss function used to evaluate the performance.
        optimizer (torch.optim.Optimizer): Optimizer used to update model weights.
        device (torch.device): Device on which to perform computations (e.g., 'cuda' or 'cpu').

    Returns:
        tuple: Returns a tuple containing the average loss and accuracy for the epoch.
    """
    model.train()  # Set the model to training mode. This is important for layers like dropout.

    running_loss = 0.0  # Total loss for the epoch.
    total = 0  # Total number of samples processed.
    correct = 0  # Total number of correct predictions.

    for inputs, labels in train_loader:
        # Move data to the appropriate device.
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()  # Clear previous gradients.

        outputs = model(inputs)  # Forward pass: compute the model output.
        # Compute the loss between the outputs and the labels.
        loss = criterion(outputs, labels)
        # Backward pass: compute the gradient of the loss wrt the model parameters.
        loss.backward()
        optimizer.step()  # Update model parameters.

        # Update running loss adjusted for the batch size.
        running_loss += loss.item() * inputs.size(0)
        # Convert outputs to binary predictions.
        preds = torch.sigmoid(outputs).data > 0.5
        # Count correct predictions.
        correct += (preds == labels.byte()).all(dim=1).sum().item()
        total += labels.size(0)  # Update total number of samples seen.

    epoch_loss = running_loss / total  # Calculate average loss over the epoch.
    epoch_acc = correct / total  # Calculate accuracy over the epoch.

    # Return the average loss and accuracy for the epoch.
    return epoch_loss, epoch_acc


def validate_model(model, valid_loader, criterion, device):
    """
    Validate the model on a validation set.

    This function evaluates the model on a validation dataset using the specified loss function and device.
    It returns the average loss, accuracy, F1 score, precision, recall, and AUC for the validation set.
    It does not update the model parameters.

    Parameters:
        model (torch.nn.Module): The model to validate.
        valid_loader (torch.utils.data.DataLoader): DataLoader for the validation data.
        criterion (torch.nn.Module): The loss function used for evaluating the model.
        device (torch.device): The device on which computations will be performed.

    Returns:
        tuple: Returns a tuple containing average loss, accuracy, F1 score, precision, recall, and AUC.
    """
    model.eval()  # Set the model to evaluation mode.
    running_loss = 0.0  # Total loss for the epoch.
    total = 0  # Total number of samples processed.
    correct = 0  # Total number of correct predictions.
    all_preds = []  # List to store all predictions.
    all_labels = []  # List to store all labels.

    # Disable gradient computation to reduce memory consumption and speed up computations.
    with torch.no_grad():
        for inputs, labels in valid_loader:
            # Move data to the appropriate device.
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)  # Compute model outputs.
            loss = criterion(outputs, labels)  # Calculate loss.

            # Update running loss adjusted for the batch size.
            running_loss += loss.item() * inputs.size(0)
            # Convert outputs to binary predictions.
            preds = torch.sigmoid(outputs).data > 0.5
            # Count correct predictions.
            correct += (preds == labels.byte()).all(dim=1).sum().item()
            total += labels.size(0)  # Update total number of samples seen.

            # Store predictions and labels for further metrics calculation.
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate average loss over the validation dataset.
    epoch_loss = running_loss / total
    # Calculate accuracy over the validation dataset.
    epoch_acc = correct / total

    # Calculate additional metrics using the collected predictions and labels.
    f1, precision, recall, auc = calculate_metrics(
        np.array(all_labels), np.array(all_preds))

    # Return validation metrics.
    return epoch_loss, epoch_acc, f1, precision, recall, auc


def train_model(model, criterion, optimizer, scheduler, train_loader, valid_loader, device, num_epochs=10, model_save=False, save_path='model.pth'):
    """
    Train and validate a model over a specified number of epochs, optionally saving the best performing model.

    This function manages the training and validation cycles for a given model, collecting performance metrics
    such as loss and accuracy, as well as F1 score, precision, recall, and AUC after each epoch. It can also save
    the model that achieves the best validation accuracy during the training process.

    Parameters:
        model (torch.nn.Module): The model to be trained and validated.
        criterion (torch.nn.Module): The loss function used to evaluate the model.
        optimizer (torch.optim.Optimizer): The optimizer used to update model parameters.
        scheduler (torch.optim.lr_scheduler): Scheduler to adjust the learning rate.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        valid_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        device (torch.device): Device on which the model will be trained (e.g., 'cuda' or 'cpu').
        num_epochs (int, optional): Number of epochs to train the model. Defaults to 10.
        model_save (bool, optional): Flag to determine whether to save the best model. Defaults to False.
        save_path (str, optional): Path where the best model will be saved. Defaults to 'model.pth'.

    Returns:
        tuple: Returns the last model state, the best model state (if saved), and a dictionary of metrics history.
    """
    best_val_acc = 0.0  # Initialize the best validation accuracy.
    best_model = None  # Placeholder for the best model.
    metrics_history = {'train_loss': [], 'train_acc': [], 'val_loss': [
    ], 'val_acc': [], 'f1': [], 'precision': [], 'recall': [], 'auc': []}
    print(f"In training loop with {num_epochs} epochs")

    for epoch in range(num_epochs):
        # Perform training for one epoch and return training loss and accuracy.
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)

        # Validate the model and return loss, accuracy, and other metrics.
        val_loss, val_acc, f1, precision, recall, auc = validate_model(
            model, valid_loader, criterion, device)

        # Update metrics history.
        metrics_history['train_loss'].append(train_loss)
        metrics_history['train_acc'].append(train_acc)
        metrics_history['val_loss'].append(val_loss)
        metrics_history['val_acc'].append(val_acc)
        metrics_history['f1'].append(f1)
        metrics_history['precision'].append(precision)
        metrics_history['recall'].append(recall)
        metrics_history['auc'].append(auc)

        # Print current epoch metrics.
        print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, AUC: {auc:.4f}')

        # Check if current model's validation accuracy is the best and save if required.
        if model_save and val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model
            save_model(model, save_path)
            print(f"Model saved with accuracy: {best_val_acc:.4f}")

        # Step the scheduler.
        scheduler.step()

    return model, best_model, metrics_history
