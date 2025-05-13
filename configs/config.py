"""Configuration parameters for sign language translator."""
import os

# Data parameters
DATA_DIR = "data/processed" # Corrected path
PROCESSED_DATA_DIR = "data/processed"
INPUT_SIZE = 128
SEQUENCE_LENGTH = 16
VALIDATION_SPLIT = 0.2 # <-- ADDED: Validation split ratio

# Model parameters
HIDDEN_SIZE = 256
DROPOUT_RATE = 0.5
NUM_LSTM_LAYERS = 2
BIDIRECTIONAL = True

# Training parameters
BATCH_SIZE = 16
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 0.0005
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 20
REDUCE_LR_PATIENCE = 8 
REDUCE_LR_FACTOR = 0.5 # Renamed from factor for clarity if needed, but keeping as is for now
NUM_WORKERS = 2 # <-- ADDED: Number of workers for DataLoader (start with 0)

# Detection parameters
MOTION_THRESHOLD = 0.002 # Default motion threshold
CONFIDENCE_THRESHOLD = 0.7 # Increased default confidence threshold
NEUTRAL_HANDICAP = 0.3     # Value to subtract from neutral class probability
HISTORY_SIZE = 5

# Paths
MODEL_SAVE_DIR = "saved_models"
# Ensure class names file path is relative to the save directory
CLASS_NAMES_FILE = os.path.join(MODEL_SAVE_DIR, "class_names.txt")
BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "best_model.pth")

# Note: The last 'import os' was redundant and has been removed.