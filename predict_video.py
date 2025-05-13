import cv2
import os
import time
import torch
import numpy as np
from torchvision import transforms
from collections import deque
import tempfile
import shutil
import mediapipe as mp # Import MediaPipe

from models import SignLanguageModel
from configs import config
from utils.preprocessing import extract_frames # Assuming this still works

# --- Global variable & Lazy Init Function ---
hands_detector_instance_pred = None

def get_hands_detector_pred():
    """Initializes and returns the MediaPipe Hands detector instance for prediction."""
    global hands_detector_instance_pred
    if hands_detector_instance_pred is None:
        print("  [MediaPipe Pred] Initializing Hands detector...")
        mp_hands = mp.solutions.hands
        hands_detector_instance_pred = mp_hands.Hands(
            static_image_mode=True, # True for processing extracted frames
            max_num_hands=2,
            min_detection_confidence=0.5
        )
        print("  [MediaPipe Pred] Hands detector initialized.")
    return hands_detector_instance_pred

def apply_mediapipe_mask_and_grayscale(image_rgb):
    """Applies MediaPipe Hands segmentation mask and converts to grayscale."""
    detector = get_hands_detector_pred() # Use the lazy init function
    image_rgb.flags.writeable = False
    results = detector.process(image_rgb)
    image_rgb.flags.writeable = True
    mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmark_points = np.array([(lm.x * image_rgb.shape[1], lm.y * image_rgb.shape[0])
                                        for lm in hand_landmarks.landmark], dtype=np.int32)
            landmark_points[:, 0] = np.clip(landmark_points[:, 0], 0, image_rgb.shape[1] - 1)
            landmark_points[:, 1] = np.clip(landmark_points[:, 1], 0, image_rgb.shape[0] - 1)
            if len(landmark_points) >= 3:
                try:
                    hull = cv2.convexHull(landmark_points)
                    cv2.fillConvexPoly(mask, hull, 255)
                except Exception as e:
                    print(f"Warning: Hull failed: {e}")
                    for point in landmark_points: cv2.circle(mask, tuple(point), 5, (255), -1)
    gray_source = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    masked_gray_image = cv2.bitwise_and(gray_source, gray_source, mask=mask)
    return masked_gray_image

def load_class_names(file_path):
    """Load class names from file."""
    if not os.path.exists(file_path):
        print(f"Error: Class names file not found at {file_path}")
        return None
    try:
        with open(file_path, 'r') as f:
            return [line.strip() for line in f.readlines()]
    except Exception as e:
        print(f"Error reading class names file {file_path}: {e}")
        return None

def load_model(model_path, num_classes, device):
    """Load a trained model (expecting 1 input channel)."""
    model = SignLanguageModel(
        num_classes=num_classes,
        input_size=config.INPUT_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        dropout_rate=config.DROPOUT_RATE,
        bidirectional=config.BIDIRECTIONAL,
        num_lstm_layers=config.NUM_LSTM_LAYERS
    ).to(device)

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None
    try:
        # Use weights_only=True if available for security
        if hasattr(torch, 'load') and 'weights_only' in torch.load.__code__.co_varnames:
             model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        else:
             model.load_state_dict(torch.load(model_path, map_location=device)) # type: ignore
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model state dict from {model_path}: {e}")
        print("Ensure the saved model corresponds to the current architecture (1 input channel).")
        return None


def capture_and_predict(duration=3):
    """Captures video, applies masking/grayscale, and predicts."""

    # --- Load Model and Classes ---
    model_path = config.BEST_MODEL_PATH
    class_names_file = config.CLASS_NAMES_FILE
    class_names = load_class_names(class_names_file)
    if class_names is None: return
    num_classes = len(class_names)
    neutral_idx = class_names.index('neutral') if 'neutral' in class_names else -1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_path, num_classes, device)
    if model is None: print("Exiting: Model failed to load."); return
    print(f"Model loaded. Using device: {device}")

    # --- Video Capture ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): print("Error: Could not open camera"); return
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, "temp_capture.mp4")
    frames_dir = os.path.join(temp_dir, "frames")

    try:
        print("Press SPACE to start recording...")
        while True:
            ret, frame = cap.read();
            if not ret: continue
            cv2.putText(frame, "Press SPACE", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Recorder', frame)
            if cv2.waitKey(1) & 0xFF == ord(' '): break

        out = cv2.VideoWriter(video_path, fourcc, 30.0, (frame_width, frame_height))
        for i in range(3, 0, -1):
            print(f"Starts in {i}...")
            start_time = time.time()
            while time.time() - start_time < 1:
                ret, frame = cap.read();
                if not ret: continue
                cv2.putText(frame, f"Start in {i}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Recorder', frame); cv2.waitKey(1)

        print(f"Recording for {duration}s...")
        start_time = time.time()
        while time.time() - start_time < duration:
            ret, frame = cap.read();
            if not ret: break
            out.write(frame)
            cv2.putText(frame, "Recording...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Recorder', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        out.release()
        print(f"Video saved temporarily.")

    finally:
        cap.release()
        cv2.destroyAllWindows()

    # --- Frame Extraction ---
    print("Extracting frames...")
    if not extract_frames(video_path, frames_dir, target_fps=10):
        print("Frame extraction failed."); shutil.rmtree(temp_dir); return
    frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    if not frame_files: print("No frames extracted."); shutil.rmtree(temp_dir); return

    # --- Preprocessing (Grayscale & Masking) ---
    print("Preprocessing frames...")
    normalize = transforms.Normalize(mean=[0.5], std=[0.5])
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.ToTensor(), # HxW -> 1xHxW
        normalize
    ])

    # --- Frame Sampling Logic ---
    sequence_length = config.SEQUENCE_LENGTH
    if len(frame_files) < sequence_length:
        frame_files += [frame_files[-1]] * (sequence_length - len(frame_files))
    elif len(frame_files) > sequence_length:
        idxs = np.linspace(0, len(frame_files) - 1, sequence_length).astype(int)
        frame_files = [frame_files[i] for i in idxs]
    # --- End Sampling ---

    frames = []
    for frame_path in frame_files:
        frame = cv2.imread(frame_path)
        if frame is None: continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- Apply Mask and Grayscale (uses lazy init now) ---
        try:
            processed_frame = apply_mediapipe_mask_and_grayscale(frame_rgb)
        except Exception as e:
            print(f"Error applying MediaPipe: {e}. Skipping frame.")
            continue
        # --- End Apply ---

        transformed_frame = transform(processed_frame) # Apply resize/normalize

        # Validate shape
        if len(transformed_frame.shape) == 2: transformed_frame = transformed_frame.unsqueeze(0)
        if transformed_frame.shape[0] != 1: transformed_frame = transformed_frame[0, :, :].unsqueeze(0)
        if transformed_frame.shape[1] != config.INPUT_SIZE or transformed_frame.shape[2] != config.INPUT_SIZE:
             resize_op = transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE), antialias=True)
             transformed_frame = resize_op(transformed_frame)
        if transformed_frame.shape[0] != 1: continue # Skip if still wrong

        frames.append(transformed_frame)

    if not frames: print("Failed to process frames."); shutil.rmtree(temp_dir); return
    # Ensure correct sequence length, padding if necessary (less ideal than sampling)
    if len(frames) < sequence_length:
        print(f"Warning: Processed {len(frames)} frames, padding to {sequence_length}.")
        padding_needed = sequence_length - len(frames)
        if frames: # Pad with last frame
            frames.extend([frames[-1]] * padding_needed)
        else: # Cannot proceed if no frames were processed
             print("Error: No frames processed after masking/transforms."); shutil.rmtree(temp_dir); return
    elif len(frames) > sequence_length: # Should not happen with sampling logic, but as safety
        frames = frames[:sequence_length]


    # --- Prediction ---
    print("Predicting sign...")
    input_tensor = torch.stack(frames).unsqueeze(0).to(device) # (1, seq_len, 1, H, W)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        # Apply Neutral Handicap
        if neutral_idx != -1 and config.NEUTRAL_HANDICAP > 0:
            probs[0, neutral_idx] = max(0.0, probs[0, neutral_idx] - config.NEUTRAL_HANDICAP)
            probs = probs / probs.sum(dim=1, keepdim=True) # Renormalize
        top_prob, top_class_idx = torch.max(probs, 1)
        predicted_class = class_names[top_class_idx.item()]
        confidence = top_prob.item()

    # --- Print Results ---
    print("-" * 30)
    print(f"Predicted Sign: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")
    print("-" * 30)
    print("Top 5 Probabilities:")
    sorted_probs, sorted_indices = torch.sort(probs[0], descending=True)
    for i in range(min(5, num_classes)):
        idx = sorted_indices[i].item()
        print(f"  - {class_names[idx]}: {sorted_probs[i].item():.4f}")

    # --- Clean up ---
    print(f"Cleaning up temporary directory: {temp_dir}")
    shutil.rmtree(temp_dir)
    # Close MediaPipe detector if it was initialized
    global hands_detector_instance_pred
    if hands_detector_instance_pred is not None:
        print("  [MediaPipe Pred] Closing Hands detector.")
        hands_detector_instance_pred.close()
        hands_detector_instance_pred = None


if __name__ == "__main__":
    capture_and_predict(duration=3)