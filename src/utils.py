import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from model import ResNet50Model

def initialize_model(device, num_classes, feature_extract=True, optimizer_name='SGD', lr=0.001, momentum=0.9, weight_decay=0.0, step_size=7, gamma=0.1):
    model = ResNet50Model(num_classes, feature_extract)
    model = model.to(device)
    
    criterion = torch.nn.BCEWithLogitsLoss()  # Suitable for multi-label classification tasks
    
    # Select the optimizer based on the input parameter
    if optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError("Unsupported optimizer")

    # Define the learning rate scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    return model, criterion, optimizer, scheduler
