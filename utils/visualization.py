"""Visualization utilities for training and evaluation."""
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os

def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path='training_history.png'):
    """Plot training and validation loss/accuracy curves."""
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    plt.grid(True)
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curves')
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path)
    print(f"Training history plot saved to {save_path}")
    
    return plt

def plot_confusion_matrix(targets, predictions, class_names, save_path='confusion_matrix.png'):
    """Plot confusion matrix of model predictions."""
    # Create the confusion matrix
    cm = confusion_matrix(targets, predictions)
    
    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")
    
    return plt

def visualize_attention(model, frame_sequence, class_names, save_path='attention.png'):
    """Visualize attention weights on frame sequence."""
    # TODO: Implement attention visualization
    pass

def create_results_directory():
    """Create a directory for storing results."""
    # Create a directory to store results
    os.makedirs('results', exist_ok=True)
    return 'results'