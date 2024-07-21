import torch

from utils import initialize_model, save_model
from train import train_model
from dataset import load_data, prepare_data_loaders
from visualize import plot_loss, plot_accuracy, plot_f1_precision_recall, plot_auc
import config

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data, all_labels = load_data()
    train_loader, valid_loader = prepare_data_loaders(data, all_labels)
    
    model, criterion, optimizer, scheduler = initialize_model(device, num_classes=len(all_labels))

    # Set save_model to True if you want to save the model after training
    trained_model, best_trained_model, metrics_history = train_model(model, criterion, optimizer, scheduler, train_loader, valid_loader, device, num_epochs=config.NUM_EPOCHS, save_model=True, save_path=config.MODEL_SAVE_PATH)

    # Save the model with the best validation accuracy
    save_model(best_trained_model, 'model_weights/final_best_model.pth')
    save_model(trained_model, 'model_weights/complete_trained_model.pth')
    
    # Plot the metrics
    save_path = config.GRAPHS_SAVE_PATH

    plot_loss(metrics_history, save_path)
    plot_accuracy(metrics_history, save_path)
    plot_f1_precision_recall(metrics_history, save_path)
    plot_auc(metrics_history, save_path)

if __name__ == '__main__':
    main()
