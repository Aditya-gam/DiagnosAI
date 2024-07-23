import matplotlib.pyplot as plt


def plot_loss(metrics_history, save_path=None):
    """
    Plot the training and validation loss per epoch and optionally save the plot to a file.

    Parameters:
        metrics_history (dict): Dictionary containing 'train_loss' and 'val_loss' lists.
        save_path (str, optional): Base path to save the plot image. The filename will be appended with '_loss.png'.

    This function creates a line plot for training and validation losses over epochs, helping visualize the model's learning progress.
    """
    plt.figure(figsize=(6, 4))
    plt.plot(metrics_history['train_loss'],
             label='Train Loss')  # Plot training loss
    # Plot validation loss
    plt.plot(metrics_history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')  # Title of the plot
    plt.xlabel('Epochs')  # X-axis label
    plt.ylabel('Loss')  # Y-axis label
    plt.legend()  # Show legend
    if save_path:  # Save the plot if a save path is provided
        plt.savefig(f"{save_path}_loss.png")
    plt.close()  # Close the plot to free up memory


def plot_accuracy(metrics_history, save_path=None):
    """
    Plot the training and validation accuracy per epoch and optionally save the plot to a file.

    Parameters:
        metrics_history (dict): Dictionary containing 'train_acc' and 'val_acc' lists.
        save_path (str, optional): Base path to save the plot image. The filename will be appended with '_accuracy.png'.

    This function creates a line plot for training and validation accuracy over epochs.
    """
    plt.figure(figsize=(6, 4))
    # Plot training accuracy
    plt.plot(metrics_history['train_acc'], label='Train Accuracy')
    # Plot validation accuracy
    plt.plot(metrics_history['val_acc'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')  # Title of the plot
    plt.xlabel('Epochs')  # X-axis label
    plt.ylabel('Accuracy')  # Y-axis label
    plt.legend()  # Show legend
    if save_path:  # Save the plot if a save path is provided
        plt.savefig(f"{save_path}_accuracy.png")
    plt.close()  # Close the plot to free up memory


def plot_f1_precision_recall(metrics_history, save_path=None):
    """
    Plot F1 score, precision, and recall per epoch and optionally save the plot to a file.

    Parameters:
        metrics_history (dict): Dictionary containing 'f1', 'precision', and 'recall' lists.
        save_path (str, optional): Base path to save the plot image. The filename will be appended with '_f1_precision_recall.png'.

    This function creates a line plot for F1 score, precision, and recall over epochs.
    """
    plt.figure(figsize=(6, 4))
    plt.plot(metrics_history['f1'], label='F1 Score')  # Plot F1 score
    plt.plot(metrics_history['precision'], label='Precision')  # Plot precision
    plt.plot(metrics_history['recall'], label='Recall')  # Plot recall
    plt.title('F1, Precision, and Recall')  # Title of the plot
    plt.xlabel('Epochs')  # X-axis label
    plt.ylabel('Score')  # Y-axis label
    plt.legend()  # Show legend
    if save_path:  # Save the plot if a save path is provided
        plt.savefig(f"{save_path}_f1_precision_recall.png")
    plt.close()  # Close the plot to free up memory


def plot_auc(metrics_history, save_path=None):
    """
    Plot AUC score per epoch and optionally save the plot to a file.

    Parameters:
        metrics_history (dict): Dictionary containing 'auc' list.
        save_path (str, optional): Base path to save the plot image. The filename will be appended with '_auc.png'.

    This function creates a line plot for AUC score over epochs.
    """
    plt.figure(figsize=(6, 4))
    plt.plot(metrics_history['auc'], label='AUC Score')  # Plot AUC score
    plt.title('AUC Score')  # Title of the plot
    plt.xlabel('Epochs')  # X-axis label
    plt.ylabel('Score')  # Y-axis label
    plt.legend()  # Show legend
    if save_path:  # Save the plot if a save path is provided
        plt.savefig(f"{save_path}_auc.png")
    plt.close()  # Close the plot to free up memory
