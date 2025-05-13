"""Main training script for the Sign Language Model."""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from tqdm import tqdm # Removed tqdm import
import numpy as np # For metrics calculation

# Local imports
from models import SignLanguageModel
from utils.data_utils import get_data_loaders
from utils.metrics import calculate_metrics # Import metrics calculation
from configs import config # Import configuration

def train_epoch(model, data_loader, criterion, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    print("    [Train Epoch] Starting...")
    # Wrap data_loader with tqdm for a progress bar - REMOVED
    # progress_bar = tqdm(data_loader, desc="Training", leave=False)
    print("    [Train Epoch] Starting batch iteration (without tqdm)...") # <-- Modified print

    # Iterate directly over data_loader
    num_batches = len(data_loader) # Get total number of batches for printing progress
    for i, (sequences, labels) in enumerate(data_loader): # <-- Iterate directly
        print(f"      [Train Epoch] Loading batch {i}...")
        sequences, labels = sequences.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * sequences.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

        # Update progress bar description - REMOVED
        # progress_bar.set_postfix(loss=loss.item())
        # Print batch loss periodically instead
        if (i + 1) % 10 == 0 or (i + 1) == num_batches: # Print every 10 batches or on the last batch
             print(f"      [Train Epoch] Batch {i+1}/{num_batches}, Loss: {loss.item():.4f}")


    print("    [Train Epoch] Finished batch iteration.")
    if total_samples == 0:
        print("    [Train Epoch] Warning: No samples processed in this epoch.")
        return 0.0, 0.0
    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    print(f"    [Train Epoch] Epoch finished. Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")
    return epoch_loss, epoch_acc

def validate_epoch(model, data_loader, criterion, device):
    """Validate the model for one epoch."""
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_predictions = []

    print("    [Val Epoch] Starting...")
    # Wrap data_loader with tqdm for a progress bar - REMOVED
    # progress_bar = tqdm(data_loader, desc="Validation", leave=False)
    print("    [Val Epoch] Starting batch iteration (without tqdm)...") # <-- Modified print

    with torch.no_grad():
        # Iterate directly over data_loader
        num_batches = len(data_loader) # Get total number of batches for printing progress
        for i, (sequences, labels) in enumerate(data_loader): # <-- Iterate directly
            print(f"      [Val Epoch] Loading batch {i}...")
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * sequences.size(0)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            # Update progress bar description - REMOVED
            # progress_bar.set_postfix(loss=loss.item())
            # Print batch loss periodically instead
            if (i + 1) % 5 == 0 or (i + 1) == num_batches: # Print every 5 batches or on the last batch
                 print(f"      [Val Epoch] Batch {i+1}/{num_batches}, Loss: {loss.item():.4f}")


    print("    [Val Epoch] Finished batch iteration.")
    num_val_samples = len(data_loader.sampler) if data_loader.sampler else len(data_loader.dataset)
    if num_val_samples == 0:
        print("    [Val Epoch] Warning: No validation samples found.")
        return 0.0, {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    epoch_loss = running_loss / num_val_samples
    metrics = calculate_metrics(all_labels, all_predictions)
    print(f"    [Val Epoch] Epoch finished. Loss: {epoch_loss:.4f}, Acc: {metrics['accuracy']:.4f}")
    return epoch_loss, metrics

def main():
    """Main training loop."""
    # Ensure saved_models directory exists
    print(f"Attempting to create directory: {config.MODEL_SAVE_DIR}")
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    print("Directory check/creation finished.")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Get data loaders and class names
    print("\nLoading data...")
    train_loader, val_loader, class_names = get_data_loaders(
        data_dir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        sequence_length=config.SEQUENCE_LENGTH,
        input_size=config.INPUT_SIZE,
        num_workers=config.NUM_WORKERS,
        validation_split=config.VALIDATION_SPLIT
    )

    # Check if data loaders were created successfully
    if train_loader is None or val_loader is None or not class_names:
        print("Error: Failed to create data loaders. Exiting.")
        return

    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")

    # Save class names
    class_names_path = config.CLASS_NAMES_FILE
    try:
        with open(class_names_path, 'w') as f:
            for name in class_names:
                f.write(f"{name}\n")
        print(f"Class names saved to {class_names_path}")
    except Exception as e:
        print(f"Error saving class names to {class_names_path}: {e}")

    # Initialize model
    print("\nInitializing model...")
    model = SignLanguageModel(
        num_classes=num_classes,
        input_size=config.INPUT_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        dropout_rate=config.DROPOUT_RATE,
        bidirectional=config.BIDIRECTIONAL,
        num_lstm_layers=config.NUM_LSTM_LAYERS
    ).to(device)
    print("Model initialized.")

    # Print model summary
    print("\nModel architecture:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss function and optimizer
    print("Defining loss function (CrossEntropyLoss)...")
    criterion = nn.CrossEntropyLoss()
    print("Defining optimizer (Adam)...")
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    # Learning rate scheduler
    print("Defining LR scheduler (ReduceLROnPlateau)...")
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config.REDUCE_LR_FACTOR, patience=config.REDUCE_LR_PATIENCE, verbose=True)

    # Training loop setup
    print("Setting up training loop variables...")
    best_val_loss = float('inf')
    epochs_no_improve = 0

    print("\nStarting training...")
    start_time = time.time()

    for epoch in range(config.NUM_EPOCHS):
        epoch_start_time = time.time()
        print(f"\n--- Epoch {epoch+1}/{config.NUM_EPOCHS} ---")

        print("  Calling train_epoch...")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device) # Calls modified function
        print("  train_epoch finished.")

        print("  Calling validate_epoch...")
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device) # Calls modified function
        print("  validate_epoch finished.")

        epoch_duration = time.time() - epoch_start_time

        # Print epoch results
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_metrics['accuracy']:.4f}")
        print(f"  Val Precision: {val_metrics['precision']:.4f} | Val Recall: {val_metrics['recall']:.4f} | Val F1: {val_metrics['f1']:.4f}")
        print(f"  Epoch Duration: {epoch_duration:.2f}s")

        # Update learning rate scheduler
        scheduler.step(val_loss)

        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            print(f"Validation loss improved ({best_val_loss:.4f} --> {val_loss:.4f}). Saving model...")
            best_val_loss = val_loss
            try:
                torch.save(model.state_dict(), config.BEST_MODEL_PATH)
                print(f"Model saved to {config.BEST_MODEL_PATH}")
                epochs_no_improve = 0 # Reset counter
            except Exception as e:
                print(f"Error saving model: {e}")
        else:
            epochs_no_improve += 1
            print(f"Validation loss did not improve. {epochs_no_improve}/{config.EARLY_STOPPING_PATIENCE}")

        # Early stopping
        if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {epochs_no_improve} epochs without improvement.")
            break

    total_training_time = time.time() - start_time
    print(f"\nTraining finished in {total_training_time // 60:.0f}m {total_training_time % 60:.0f}s")
    print(f"Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()