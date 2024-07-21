import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from model import ResNet50Model


def initialize_model(device, num_classes, feature_extract=True, optimizer_name='SGD', lr=0.001, momentum=0.9, weight_decay=0.0, step_size=7, gamma=0.1):
    model = ResNet50Model(num_classes, feature_extract)
    model = model.to(device)

    # Suitable for multi-label classification tasks
    criterion = torch.nn.BCEWithLogitsLoss()

    # Select the optimizer based on the input parameter
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
        raise ValueError("Unsupported optimizer")

    # Define the learning rate scheduler
    scheduler = lr_scheduler.StepLR(
        optimizer, step_size=step_size, gamma=gamma)

    return model, criterion, optimizer, scheduler


def save_model(model, path):
    """ Saves the model's state_dict to the specified path if the model has improved based on accuracy """
    torch.save(model.state_dict(), path)


def create_metrics_list(metrics_history, train_loss, train_acc, val_loss, val_acc, f1, precision, recall, auc):
    # Save metrics history for graphing
    metrics_history['train_loss'].append(train_loss)
    metrics_history['train_acc'].append(train_acc)
    metrics_history['val_loss'].append(val_loss)
    metrics_history['val_acc'].append(val_acc)
    metrics_history['f1'].append(f1)
    metrics_history['precision'].append(precision)
    metrics_history['recall'].append(recall)
    metrics_history['auc'].append(auc)

    return metrics_history
