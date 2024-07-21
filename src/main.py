import torch

from utils import initialize_model
from train import train_model
from dataset import load_data, prepare_data_loaders

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data, all_labels = load_data()
    train_loader, valid_loader = prepare_data_loaders(data, all_labels)
    
    model, criterion, optimizer, scheduler = initialize_model(device, num_classes=len(all_labels))

    # Set save_model to True if you want to save the model after training
    trained_model, metrics_history = train_model(model, criterion, optimizer, scheduler, train_loader, valid_loader, device, num_epochs=25, save_model=True, save_path='best_model.pth')

    # Now you can use metrics_history for graphing or further analysis

if __name__ == '__main__':
    main()
