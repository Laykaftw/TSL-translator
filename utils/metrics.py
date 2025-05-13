"""Evaluation metrics calculation."""

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np

def calculate_metrics(y_true, y_pred, average='weighted'):
    """
    Calculates classification metrics.

    Args:
        y_true: List or array of true labels.
        y_pred: List or array of predicted labels.
        average: Type of averaging for precision, recall, f1 ('weighted', 'macro', 'micro', None).
                 'weighted' accounts for label imbalance.

    Returns:
        A dictionary containing accuracy, precision, recall, and f1-score.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def calculate_confusion_matrix(y_true, y_pred, class_names=None):
    """
    Calculates the confusion matrix.

    Args:
        y_true: List or array of true labels.
        y_pred: List or array of predicted labels.
        class_names: Optional list of class names for labeling the matrix axes.

    Returns:
        A numpy array representing the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)) if class_names else None)
    return cm

# Example usage (optional)
if __name__ == '__main__':
    true_labels = [0, 1, 2, 0, 1, 2, 0, 2, 1, 1]
    pred_labels = [0, 2, 1, 0, 1, 2, 0, 2, 1, 0]
    class_names_example = ['Class A', 'Class B', 'Class C']

    print("Example Metrics Calculation:")
    metrics = calculate_metrics(true_labels, pred_labels, average='weighted')
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision (weighted): {metrics['precision']:.4f}")
    print(f"  Recall (weighted): {metrics['recall']:.4f}")
    print(f"  F1-Score (weighted): {metrics['f1']:.4f}")

    metrics_macro = calculate_metrics(true_labels, pred_labels, average='macro')
    print(f"\n  Precision (macro): {metrics_macro['precision']:.4f}")
    print(f"  Recall (macro): {metrics_macro['recall']:.4f}")
    print(f"  F1-Score (macro): {metrics_macro['f1']:.4f}")

    print("\nExample Confusion Matrix:")
    cm = calculate_confusion_matrix(true_labels, pred_labels, class_names=class_names_example)
    print(cm)

    # You might want to visualize the confusion matrix using libraries like seaborn or matplotlib
    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # import pandas as pd
    # cm_df = pd.DataFrame(cm, index=class_names_example, columns=class_names_example)
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    # plt.title('Confusion Matrix')
    # plt.ylabel('Actual Label')
    # plt.xlabel('Predicted Label')
    # plt.show()