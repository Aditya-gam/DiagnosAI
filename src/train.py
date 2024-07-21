import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = torch.sigmoid(outputs).data > 0.5  # Threshold predictions for multi-label
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
    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    auc = roc_auc_score(all_labels, all_preds, average='macro')

    return epoch_loss, epoch_acc, f1, precision, recall, auc

def train_model(model, criterion, optimizer, scheduler, train_loader, valid_loader, device, num_epochs=25, save_model=False, save_path='model.pth'):
    best_val_acc = 0.0
    metrics_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'f1': [], 'precision': [], 'recall': [], 'auc': []}

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, f1, precision, recall, auc = validate_model(model, valid_loader, criterion, device)

        # Save metrics history for graphing
        metrics_history['train_loss'].append(train_loss)
        metrics_history['train_acc'].append(train_acc)
        metrics_history['val_loss'].append(val_loss)
        metrics_history['val_acc'].append(val_acc)
        metrics_history['f1'].append(f1)
        metrics_history['precision'].append(precision)
        metrics_history['recall'].append(recall)
        metrics_history['auc'].append(auc)

        print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, AUC: {auc:.4f}')

        if save_model and val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"Model saved with accuracy: {best_val_acc:.4f}")

        scheduler.step()
    
    return model, metrics_history