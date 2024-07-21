from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def calculate_metrics(labels, predictions):
    try:
        auc = roc_auc_score(labels, predictions,
                            average='macro', multi_class='ovr')
    except ValueError:
        auc = float('nan')  # Not enough classes to calculate AUC
        print("Not enough classes to calculate AUC. Returning NaN.")

    # Other metrics
    f1 = f1_score(labels, predictions, average='macro')
    precision = precision_score(
        labels, predictions, average='macro', zero_division=0)
    recall = recall_score(labels, predictions,
                          average='macro', zero_division=0)

    return f1, precision, recall, auc
