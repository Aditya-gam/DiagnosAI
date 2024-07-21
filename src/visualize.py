import matplotlib.pyplot as plt

def plot_loss(metrics_history, save_path=None):
    plt.figure(figsize=(6, 4))
    plt.plot(metrics_history['train_loss'], label='Train Loss')
    plt.plot(metrics_history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    if save_path:
        plt.savefig(f"{save_path}_loss.png")
    plt.close()

def plot_accuracy(metrics_history, save_path=None):
    plt.figure(figsize=(6, 4))
    plt.plot(metrics_history['train_acc'], label='Train Accuracy')
    plt.plot(metrics_history['val_acc'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    if save_path:
        plt.savefig(f"{save_path}_accuracy.png")
    plt.close()

def plot_f1_precision_recall(metrics_history, save_path=None):
    plt.figure(figsize=(6, 4))
    plt.plot(metrics_history['f1'], label='F1 Score')
    plt.plot(metrics_history['precision'], label='Precision')
    plt.plot(metrics_history['recall'], label='Recall')
    plt.title('F1, Precision, and Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    if save_path:
        plt.savefig(f"{save_path}_f1_precision_recall.png")
    plt.close()

def plot_auc(metrics_history, save_path=None):
    plt.figure(figsize=(6, 4))
    plt.plot(metrics_history['auc'], label='AUC Score')
    plt.title('AUC Score')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    if save_path:
        plt.savefig(f"{save_path}_auc.png")
    plt.close()
