import torch

from utils import initialize_model
from train import train_model
from dataset import load_data, prepare_data_loaders

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data, all_labels = load_data()
    train_loader, valid_loader = prepare_data_loaders(data, all_labels)
    
    model, criterion, optimizer, scheduler = initialize_model(device, num_classes=len(all_labels))
    
    trained_model = train_model(model, criterion, optimizer, scheduler, train_loader, valid_loader, device, num_epochs=25)
    # Additional evaluation can be performed here

if __name__ == '__main__':
    main()
