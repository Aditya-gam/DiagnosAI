import torch

from utils import initialize_model, save_model, count_parameters
from train import train_model
from dataset import load_data, prepare_data_loaders, prepare_test_loader
from visualize import plot_loss, plot_accuracy, plot_f1_precision_recall, plot_auc
import config


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_val_data, test_data, all_labels = load_data()
    train_loader, valid_loader = prepare_data_loaders(
        train_val_data, all_labels)
    print(f"Number of training samples: {len(train_loader.dataset)}")
    print(f"Number of validation samples: {len(valid_loader.dataset)}")

    test_loader = prepare_test_loader(test_data, all_labels)
    print(f"Number of test samples: {len(test_loader.dataset)}")

    model, criterion, optimizer, scheduler = initialize_model(device, num_classes=len(all_labels), optimizer_name='AdamW', lr=0.001, weight_decay=0.01, step_size=7, gamma=0.1)
    total_params, trainable_params, non_trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Non-trainable parameters: {non_trainable_params}")

    trained_model, best_trained_model, metrics_history = train_model(
        model, criterion, optimizer, scheduler, train_loader, valid_loader, device, num_epochs=config.NUM_EPOCHS, model_save=True, save_path=config.MODEL_SAVE_PATH)

    # Save the model with the best validation accuracy
    save_model(best_trained_model, 'model_weights/final_best_model.pth')
    print("best_trained_model saved at: model_weights/final_best_model.pth")
    save_model(trained_model, 'model_weights/complete_trained_model.pth')
    print("trained_model saved at: model_weights/complete_trained_model.pth")

    # Plot the metrics
    save_path = config.GRAPHS_SAVE_PATH

    plot_loss(metrics_history, save_path)
    plot_accuracy(metrics_history, save_path)
    plot_f1_precision_recall(metrics_history, save_path)
    plot_auc(metrics_history, save_path)


if __name__ == '__main__':
    main()
