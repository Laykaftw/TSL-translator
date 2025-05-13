"""Real-time sign language detection with MediaPipe masking and grayscale (Lazy Init)."""
import cv2
import torch
import numpy as np
from torchvision import transforms
from collections import deque
import os
import time
import mediapipe as mp # Import MediaPipe

from models import SignLanguageModel
from configs import config # Import config directly

# --- Global variable & Lazy Init Function ---
hands_detector_instance_rt = None

def get_hands_detector_rt():
    """Initializes and returns the MediaPipe Hands detector instance for real-time."""
    global hands_detector_instance_rt
    if hands_detector_instance_rt is None:
        print("  [MediaPipe RT] Initializing Hands detector...")
        mp_hands = mp.solutions.hands
        hands_detector_instance_rt = mp_hands.Hands(
            static_image_mode=False, # False for real-time tracking
            max_num_hands=2,
            min_detection_confidence=0.6, # Confidence for detection
            min_tracking_confidence=0.5  # Confidence for tracking
        )
        print("  [MediaPipe RT] Hands detector initialized.")
    return hands_detector_instance_rt

def apply_mediapipe_mask_and_grayscale(image_rgb):
    """Applies MediaPipe Hands segmentation mask and converts to grayscale."""
    detector = get_hands_detector_rt() # Use the lazy init function for real-time
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
    return masked_gray_image, mask # Return mask for visualization

def load_class_names(file_path):
    """Load class names from file."""
    if not os.path.exists(file_path): return None
    try:
        with open(file_path, 'r') as f: return [line.strip() for line in f.readlines()]
    except Exception as e: print(f"Error reading class names: {e}"); return None

def load_model(model_path, num_classes, device):
    """Load a trained model (expecting 1 input channel)."""
    model = SignLanguageModel(num_classes=num_classes, input_size=config.INPUT_SIZE, hidden_size=config.HIDDEN_SIZE, dropout_rate=config.DROPOUT_RATE, bidirectional=config.BIDIRECTIONAL, num_lstm_layers=config.NUM_LSTM_LAYERS).to(device)
    if not os.path.exists(model_path): print(f"Error: Model not found at {model_path}"); return None
    try:
        if hasattr(torch, 'load') and 'weights_only' in torch.load.__code__.co_varnames:
             model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        else:
             model.load_state_dict(torch.load(model_path, map_location=device)) # type: ignore
        model.eval(); return model
    except Exception as e: print(f"Error loading model: {e}"); return None


def real_time_detection():
    """Run real-time detection from webcam with masking."""
    model_path = config.BEST_MODEL_PATH
    class_names_file = config.CLASS_NAMES_FILE
    class_names = load_class_names(class_names_file)
    if class_names is None: return
    num_classes = len(class_names)
    neutral_idx = class_names.index('neutral') if 'neutral' in class_names else -1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_path, num_classes, device)
    if model is None: return
    print(f"Model loaded. Using device: {device}")

    cap = None
    for idx in [0, 1, 2]:
        try:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                break
            else:
                cap.release()
        except Exception as e:
            print(f"Error opening camera index {idx}: {e}")
    if cap is None or not cap.isOpened():
        print("Failed to open camera.")
        return

    # Transforms for grayscale input
    normalize = transforms.Normalize(mean=[0.5], std=[0.5])
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.ToTensor(), # HxW -> 1xHxW
        normalize
    ])

    # Buffers and thresholds
    frame_buffer = deque(maxlen=config.SEQUENCE_LENGTH)
    prediction_history = deque(maxlen=config.HISTORY_SIZE)
    motion_threshold = config.MOTION_THRESHOLD # May need adjustment
    confidence_threshold = config.CONFIDENCE_THRESHOLD

    # Motion detection (on original frame ROI)
    prev_frame_gray = None
    motion_history = deque(maxlen=5) # Average over last 5 frames

    print("Starting real-time detection (Grayscale & Masking - Lazy Init). Press 'q' to quit.")
    print(f"Motion Threshold: {motion_threshold:.6f} (+/- to adjust)")
    print(f"Confidence Threshold: {confidence_threshold:.2f}")
    print(f"Sequence Length: {config.SEQUENCE_LENGTH}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break

            frame_height, frame_width, _ = frame.shape
            display_frame = frame.copy() # For showing original + overlays

            # --- Motion Detection (on original frame ROI) ---
            roi_x, roi_y = int(frame_width * 0.05), int(frame_height * 0.05)
            roi_w, roi_h = int(frame_width * 0.9), int(frame_height * 0.9)
            # Ensure ROI dimensions are valid
            if roi_w <= 0 or roi_h <= 0: continue
            roi_bgr = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            if roi_bgr.size == 0: continue # Skip if ROI is empty
            roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

            avg_motion = 0
            if prev_frame_gray is not None and prev_frame_gray.shape == roi_gray.shape:
                frame_diff = cv2.absdiff(roi_gray, prev_frame_gray)
                # Normalize motion score by area and max pixel value
                motion_score = np.sum(frame_diff) / (roi_w * roi_h * 255.0)
                motion_history.append(motion_score)
                avg_motion = sum(motion_history) / len(motion_history) if motion_history else 0
            prev_frame_gray = roi_gray
            # --- End Motion Detection ---

            # --- Preprocessing: Masking & Grayscaling ---
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                # Calls lazy init function internally
                processed_frame, mask_vis = apply_mediapipe_mask_and_grayscale(frame_rgb)
            except Exception as e:
                print(f"Error in MediaPipe processing: {e}")
                continue # Skip frame

            transformed_frame = transform(processed_frame) # Apply resize/normalize

            # Validate shape
            if len(transformed_frame.shape) == 2: transformed_frame = transformed_frame.unsqueeze(0)
            if transformed_frame.shape[0] != 1: transformed_frame = transformed_frame[0, :, :].unsqueeze(0)
            if transformed_frame.shape[1] != config.INPUT_SIZE or transformed_frame.shape[2] != config.INPUT_SIZE:
                 resize_op = transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE), antialias=True)
                 transformed_frame = resize_op(transformed_frame)
            if transformed_frame.shape[0] != 1: continue # Skip if still wrong

            frame_buffer.append(transformed_frame)
            # --- End Preprocessing ---

            # --- Prediction Logic ---
            text = f"Collecting... ({len(frame_buffer)}/{config.SEQUENCE_LENGTH})"
            confidence_score = 0; predicted_class = None; probabilities = None
            predicted_idx = -1 # Initialize predicted index

            # Trigger prediction only when buffer is full AND motion is detected
            trigger_prediction = (len(frame_buffer) == config.SEQUENCE_LENGTH) and \
                                 (avg_motion > motion_threshold)

            if trigger_prediction:
                input_tensor = torch.stack(list(frame_buffer)).unsqueeze(0).to(device) # (1, seq, 1, H, W)
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    # Apply Neutral Handicap
                    if neutral_idx != -1 and config.NEUTRAL_HANDICAP > 0:
                        probs[0, neutral_idx] = max(0.0, probs[0, neutral_idx] - config.NEUTRAL_HANDICAP)
                        probs = probs / probs.sum(dim=1, keepdim=True) # Renormalize
                    top_prob, top_class_idx = torch.max(probs, 1)
                    predicted_idx = top_class_idx.item()
                    confidence = top_prob.item()
                    probabilities = probs[0].cpu().numpy()

                confidence_score = confidence * 100
                if confidence > confidence_threshold:
                    predicted_class = class_names[predicted_idx]
                    prediction_history.append((predicted_idx, confidence_score))
                    # Temporal Smoothing: Check if last N predictions are consistent
                    if len(prediction_history) >= 2: # Check last 2 predictions
                        pred_counts = {}
                        for p_idx, _ in prediction_history:
                            pred_counts[p_idx] = pred_counts.get(p_idx, 0) + 1
                        most_common = max(pred_counts.items(), key=lambda x: x[1])
                        # Require at least 2 consecutive or recent same predictions
                        if most_common[1] >= 2:
                            text = f"Pred: {class_names[most_common[0]]} ({confidence_score:.1f}%)"
                        else:
                            text = "Uncertain" # Not stable yet
                    else: # First prediction above threshold
                        text = f"Detect: {predicted_class} ({confidence_score:.1f}%)"
                else:
                    text = f"Low conf: {confidence_score:.1f}%"
                    prediction_history.clear() # Clear history if confidence drops

            elif len(frame_buffer) == config.SEQUENCE_LENGTH:
                 # Buffer is full but no motion detected
                 text = f"No motion"
                 prediction_history.clear() # Clear history if motion stops
            # --- End Prediction Logic ---

            # --- Display ---
            # Overlay mask for visualization (optional)
            mask_colored = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)
            overlay = cv2.addWeighted(display_frame, 0.7, mask_colored, 0.3, 0)

            # Display prediction text
            cv2.putText(overlay, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # Display motion score
            motion_text = f"Motion: {avg_motion:.6f} (Th: {motion_threshold:.6f})"
            cv2.putText(overlay, motion_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            # Display buffer status
            buffer_status = f"Buffer: {len(frame_buffer)}/{config.SEQUENCE_LENGTH}"
            cv2.putText(overlay, buffer_status, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Display top 5 probabilities
            if probabilities is not None:
                bar_height=15; bar_width=100; bar_gap=5; start_y=frame_height-(5*(bar_height+bar_gap))-10
                sorted_indices = np.argsort(probabilities)[::-1]
                for i, idx in enumerate(sorted_indices[:5]):
                    label_text = f"{class_names[idx]}: {probabilities[idx]*100:.1f}%"
                    text_y = start_y + i*(bar_height+bar_gap)
                    cv2.putText(overlay, label_text, (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    bar_length = int(probabilities[idx] * bar_width)
                    # Highlight the top prediction if above threshold
                    color = (0, 255, 0) if idx == predicted_idx and confidence > confidence_threshold else (0, 165, 255) # Green if confident, Orange otherwise
                    cv2.rectangle(overlay, (150, text_y-bar_height+3), (150+bar_length, text_y+3), color, -1)

            cv2.imshow('Sign Language Detection (Masked)', overlay)
            # --- End Display ---

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('+') or key == ord('='): # Increase threshold
                motion_threshold *= 1.2
                print(f"Motion Th: {motion_threshold:.6f}")
            elif key == ord('-') or key == ord('_'): # Decrease threshold
                motion_threshold /= 1.2
                print(f"Motion Th: {motion_threshold:.6f}")

    except KeyboardInterrupt: print("\nDetection interrupted.")
    except Exception as e: print(f"Error during detection: {e}"); import traceback; traceback.print_exc()
    finally:
        if cap is not None: cap.release()
        cv2.destroyAllWindows()
        # Close MediaPipe detector if it was initialized
        global hands_detector_instance_rt
        if hands_detector_instance_rt is not None:
            print("  [MediaPipe RT] Closing Hands detector.")
            hands_detector_instance_rt.close()
            hands_detector_instance_rt = None
        print("Detection stopped.")


def main():
    real_time_detection()

if __name__ == "__main__":
    main()