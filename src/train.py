import torch
import numpy as np
from utils import save_model, create_metrics_list
from evaluate import calculate_metrics

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = torch.sigmoid(outputs).data > 0.5
        correct += (preds == labels.byte()).all(dim=1).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc

def validate_model(model, valid_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total = 0
    correct = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            preds = torch.sigmoid(outputs).data > 0.5
            correct += (preds == labels.byte()).all(dim=1).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    f1, precision, recall, auc = calculate_metrics(np.array(all_labels), np.array(all_preds))

    return epoch_loss, epoch_acc, f1, precision, recall, auc

def train_model(model, criterion, optimizer, scheduler, train_loader, valid_loader, device, num_epochs=25, model_save=False, save_path='model.pth'):
    best_val_acc = 0.0
    best_model = None
    metrics_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'f1': [], 'precision': [], 'recall': [], 'auc': []}

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, f1, precision, recall, auc = validate_model(model, valid_loader, criterion, device)

        # Save metrics history for graphing
        metrics_history = create_metrics_list(metrics_history, train_loss, train_acc, val_loss, val_acc, f1, precision, recall, auc)

        print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, AUC: {auc:.4f}')

        if model_save and val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model.state_dict()
            save_model(model, save_path, best_val_acc)

        scheduler.step()
    
    return model, best_model , metrics_history