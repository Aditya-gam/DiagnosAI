from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def calculate_metrics(labels, predictions):
    """ Calculate and return the F1 score, precision, recall, and AUC-ROC for the given labels and predictions """
    f1 = f1_score(labels, predictions, average='macro')
    precision = precision_score(labels, predictions, average='macro')
    recall = recall_score(labels, predictions, average='macro')
    auc = roc_auc_score(labels, predictions, average='macro')

    return f1, precision, recall, auc
