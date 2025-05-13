"""Evaluate the trained Sign Language Model on the validation set."""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns # For plotting confusion matrix
import matplotlib.pyplot as plt # For plotting confusion matrix
import pandas as pd # For displaying confusion matrix nicely

# Local imports
from models import SignLanguageModel
from utils.data_utils import get_data_loaders # To get the validation loader
from configs import config # Import configuration

def evaluate_model(model, data_loader, device, class_names):
    """Evaluate the model on the given data loader."""
    model.eval() # Set model to evaluation mode
    all_labels = []
    all_predictions = []

    print("Starting evaluation...")
    print(f"Using device: {device}")

    with torch.no_grad(): # No need to track gradients during evaluation
        # Iterate directly over data_loader (no tqdm needed here unless dataset is huge)
        num_batches = len(data_loader)
        for i, (sequences, labels) in enumerate(data_loader):
            print(f"  Processing batch {i+1}/{num_batches}...")
            sequences, labels = sequences.to(device), labels.to(device)

            outputs = model(sequences)
            _, predicted = torch.max(outputs.data, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    print("Evaluation finished.")

    # Calculate metrics
    print("\n--- Evaluation Results ---")
    accuracy = accuracy_score(all_labels, all_predictions)
    print(f"Overall Accuracy: {accuracy:.4f}")

    # Classification Report (Precision, Recall, F1-score per class)
    print("\nClassification Report:")
    # Use zero_division=0 to handle cases where a class might have no predictions
    report = classification_report(all_labels, all_predictions, target_names=class_names, zero_division=0)
    print(report)

    # Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_predictions)

    # Display Confusion Matrix using Pandas for better readability in console
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print(cm_df)

    # Optional: Plot and save confusion matrix as an image
    try:
        plt.figure(figsize=(15, 12)) # Adjust size as needed for number of classes
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual Class')
        plt.xlabel('Predicted Class')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        cm_save_path = os.path.join(config.MODEL_SAVE_DIR, "confusion_matrix.png")
        plt.savefig(cm_save_path)
        print(f"\nConfusion matrix plot saved to {cm_save_path}")
        # plt.show() # Uncomment to display the plot directly if running in an interactive environment
    except Exception as e:
        print(f"\nCould not plot confusion matrix: {e}")

    return accuracy, report, cm

def main():
    """Main evaluation function."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Load Class Names ---
    class_names_path = config.CLASS_NAMES_FILE # From config
    try:
        with open(class_names_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        num_classes = len(class_names)
        print(f"Loaded {num_classes} classes from {class_names_path}")
    except FileNotFoundError:
        print(f"Error: Class names file not found at {class_names_path}")
        return
    except Exception as e:
        print(f"Error reading class names: {e}")
        return

    # --- Get Validation Data Loader ---
    # We only need the validation loader, so we can ignore the train loader
    print("\nLoading validation data...")
    # Use num_workers=0 for evaluation simplicity unless it's very slow
    # Set shuffle=False for validation/evaluation consistency
    _, val_loader, _ = get_data_loaders(
        data_dir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE, # Use configured batch size or a larger one if memory allows
        sequence_length=config.SEQUENCE_LENGTH,
        input_size=config.INPUT_SIZE,
        num_workers=0, # Use 0 for simplicity here
        validation_split=config.VALIDATION_SPLIT,
        shuffle=False # Important: Don't shuffle validation data for consistent evaluation
    )

    if val_loader is None:
        print("Error: Failed to create validation data loader. Exiting.")
        return
    print("Validation data loader created.")

    # --- Initialize Model ---
    print("\nInitializing model structure...")
    model = SignLanguageModel(
        num_classes=num_classes, # Use loaded number of classes
        input_size=config.INPUT_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        dropout_rate=config.DROPOUT_RATE,
        bidirectional=config.BIDIRECTIONAL,
        num_lstm_layers=config.NUM_LSTM_LAYERS
    )

    # --- Load Trained Weights ---
    model_path = config.BEST_MODEL_PATH # From config
    print(f"Loading trained weights from {model_path}")
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Weights loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Trained model file not found at {model_path}")
        return
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    model = model.to(device) # Move model to the appropriate device

    # --- Run Evaluation ---
    evaluate_model(model, val_loader, device, class_names)

if __name__ == "__main__":
    main()