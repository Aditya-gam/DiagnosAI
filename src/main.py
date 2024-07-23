import torch

from utils import initialize_model, save_model, count_parameters
from train import train_model
from dataset import load_data, prepare_data_loaders, prepare_test_loader
from visualize import plot_loss, plot_accuracy, plot_f1_precision_recall, plot_auc
import config


def main():
    """
    Main function to execute the training and evaluation pipeline for a chest X-ray classification model.

    This function orchestrates the data loading, model initialization, training, and evaluation processes. It uses
    custom modules to load data, prepare DataLoader objects, initialize and count model parameters, train the model,
    save the trained model, and plot training metrics. It also handles device allocation to utilize GPU if available.
    """

    # Set the device to GPU if available, otherwise use CPU.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the data and prepare DataLoader objects for training, validation, and testing.
    train_val_data, test_data, all_labels = load_data()
    train_loader, valid_loader = prepare_data_loaders(
        train_val_data, all_labels)
    print(f"Number of training samples: {len(train_loader.dataset)}")
    print(f"Number of validation samples: {len(valid_loader.dataset)}")

    test_loader = prepare_test_loader(test_data, all_labels)
    print(f"Number of test samples: {len(test_loader.dataset)}")

    # Initialize the model, loss criterion, optimizer, and learning rate scheduler.
    model, criterion, optimizer, scheduler = initialize_model(
        device, num_classes=len(all_labels), optimizer_name='AdamW',
        lr=0.001, weight_decay=0.01, step_size=7, gamma=0.1
    )

    # Count the total, trainable, and non-trainable parameters in the model.
    total_params, trainable_params, non_trainable_params = count_parameters(
        model)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Non-trainable parameters: {non_trainable_params}")

    # Train the model and return the best and last model states along with their training metrics history.
    trained_model, best_trained_model, metrics_history = train_model(
        model, criterion, optimizer, scheduler, train_loader, valid_loader,
        device, num_epochs=config.NUM_EPOCHS, model_save=True, save_path=config.MODEL_SAVE_PATH
    )

    # Save the best and final models to disk.
    save_model(best_trained_model, 'model_weights/final_best_model.pth')
    print("best_trained_model saved at: model_weights/final_best_model.pth")
    save_model(trained_model, 'model_weights/complete_trained_model.pth')
    print("trained_model saved at: model_weights/complete_trained_model.pth")

    # Plot the metrics using the recorded history and save the plots to the configured path.
    save_path = config.GRAPHS_SAVE_PATH
    plot_loss(metrics_history, save_path)
    plot_accuracy(metrics_history, save_path)
    plot_f1_precision_recall(metrics_history, save_path)
    plot_auc(metrics_history, save_path)


# Ensure that the main function is run only when this script is executed directly, not when imported.
if __name__ == '__main__':
    main()
