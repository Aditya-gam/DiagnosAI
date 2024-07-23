from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def calculate_metrics(labels, predictions):
    """
    Calculate and return key classification metrics including F1 score, precision, recall, and AUC.

    This function computes the F1 score, precision, recall, and AUC for the given predictions
    compared to the true labels. It handles multi-class classification scenarios by calculating
    metrics in a macro-averaged format. The function is robust to exceptions in AUC calculation
    when insufficient classes are present.

    Parameters:
        labels (array-like): True labels for the data.
        predictions (array-like): Predicted labels by the model.

    Returns:
        tuple: Returns a tuple containing the F1 score, precision, recall, and AUC. If AUC cannot be calculated,
               it returns NaN for AUC and prints a warning message.

    Notes:
        - `average='macro'` calculates metrics independently for each class and then takes the average, treating all classes equally.
        - `multi_class='ovr'` (one-vs-rest) is used for AUC to handle multi-class classification by considering each class against all others.
        - `zero_division=0` argument in precision and recall functions handles cases where a division by zero occurs, returning 0 in such cases.
    """
    try:
        # Attempt to calculate the AUC using a macro-average and one-vs-rest approach for multi-class classification.
        auc = roc_auc_score(labels, predictions,
                            average='macro', multi_class='ovr')
    except ValueError:
        # Handle cases where AUC cannot be calculated due to insufficient number of classes
        auc = float('nan')  # Assign NaN to AUC if calculation fails.
        print("Not enough classes to calculate AUC. Returning NaN.")

    # Calculate other classification metrics using macro averaging.
    f1 = f1_score(labels, predictions, average='macro')  # Calculate F1 score.
    # Calculate precision.
    precision = precision_score(
        labels, predictions, average='macro', zero_division=0)
    # Calculate recall.
    recall = recall_score(labels, predictions,
                          average='macro', zero_division=0)

    # Return the calculated metrics as a tuple.
    return f1, precision, recall, auc
