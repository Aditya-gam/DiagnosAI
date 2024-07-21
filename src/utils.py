import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from model import ResNet50Model

def initialize_model(device, num_classes, feature_extract=True, lr=0.001, step_size=7, gamma=0.1):
    model = ResNet50Model(num_classes, feature_extract)
    model = model.to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    return model, criterion, optimizer, scheduler
