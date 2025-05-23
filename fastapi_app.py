# fastapi_server.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from torchvision import transforms
import cv2
import numpy as np
import mediapipe as mp
import torch
import os
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SERVER_DEBUG_ROOT_DIR = "server_side_debug_frames"
os.makedirs(SERVER_DEBUG_ROOT_DIR, exist_ok=True)

try:
    from configs import config
    from models.model import SignLanguageModel
except ImportError as e:
    logger.error(f"Failed to import local modules. Error: {e}")
    class ConfigFallback:
        INPUT_SIZE = 128
        SEQUENCE_LENGTH = 16
        HIDDEN_SIZE = 256
        DROPOUT_RATE = 0.5
        BIDIRECTIONAL = True
        NUM_LSTM_LAYERS = 2
        MODEL_SAVE_DIR = "saved_models"
        CLASS_NAMES_FILE = os.path.join(MODEL_SAVE_DIR, "class_names.txt")
        BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "best_model.pth")
    config = ConfigFallback()

    if 'SignLanguageModel' not in globals():
        class SignLanguageModel(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.fc = torch.nn.Linear(1, 1)
                logger.info("[Model Init Fallback] Dummy model initialized.")

            def forward(self, x):
                return self.fc(torch.randn(x.size(0), 1))

# --- Global Variables ---
hands_detector_instance = None
model_loaded = None
class_names_loaded = None
device_loaded = None
transform_loaded = None

def get_hands_detector():
    global hands_detector_instance
    if hands_detector_instance is None:
        logger.info("Server: Initializing MediaPipe Hands...")
        mp_hands = mp.solutions.hands
        hands_detector_instance = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.2,  # Lowered from 0.5 for better detection
            min_tracking_confidence=0.2,   # Lowered from 0.4 for better tracking
            model_complexity=0             # Use higher complexity model (0, 1)
        )
        logger.info("Server: MediaPipe Hands detector initialized with improved settings.")
    return hands_detector_instance

def get_rotation_type():
    # Try different rotations to match your laptop webcam training data:
    # - cv2.ROTATE_90_CLOCKWISE: Portrait → Landscape (normal case)
    # - cv2.ROTATE_90_COUNTERCLOCKWISE: Portrait → Landscape (alternative)
    # - None: No rotation (if mobile is already landscape)
    
    rotation_type = cv2.ROTATE_180  # Try this if clockwise didn't work
    logger.info(f"Applying rotation to match laptop training data: {rotation_type}")
    return rotation_type

def create_fallback_hand_mask(image_rgb: np.ndarray) -> np.ndarray:
    """Create a basic hand mask using skin color detection as fallback."""
    try:
        # Convert to HSV for skin detection
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        
        # Define skin color range in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create mask for skin color
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Remove noise and fill gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        
        # Find largest contour (likely to be hand)
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 1000:  # Minimum area threshold
                mask = np.zeros_like(skin_mask)
                cv2.fillPoly(mask, [largest_contour], 255)
                return mask
        
        return np.zeros_like(skin_mask)
    except Exception as e:
        logger.warning(f"Fallback mask creation failed: {e}")
        return np.zeros((image_rgb.shape[0], image_rgb.shape[1]), dtype=np.uint8)

def apply_mediapipe_mask_and_grayscale_internal(image_rgb: np.ndarray, last_valid_mask=None) -> tuple:
    """Apply MediaPipe masking + grayscale conversion with improved hand detection."""
    detector = get_hands_detector()
    
    # Enhance image for better detection
    enhanced_image = image_rgb.copy()
    
    # Apply histogram equalization to improve contrast
    lab = cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2LAB)
    lab[:,:,0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(lab[:,:,0])
    enhanced_image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # Try detection on enhanced image first
    enhanced_image.flags.writeable = False
    results = detector.process(enhanced_image)
    
    # If no hands found, try on original image with different preprocessing
    if not results.multi_hand_landmarks:
        # Try with brightness adjustment
        bright_image = cv2.convertScaleAbs(image_rgb, alpha=1.2, beta=20)
        bright_image.flags.writeable = False
        results = detector.process(bright_image)
        
        # If still no hands, try with blur reduction (sharpening)
        if not results.multi_hand_landmarks:
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharp_image = cv2.filter2D(image_rgb, -1, kernel)
            sharp_image.flags.writeable = False
            results = detector.process(sharp_image)
    
    enhanced_image.flags.writeable = True

    mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    has_hands = False
    
    if results.multi_hand_landmarks:
        has_hands = True
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks_px = np.array(
                [(int(lm.x * image_rgb.shape[1]), int(lm.y * image_rgb.shape[0]))
                 for lm in hand_landmarks.landmark],
                dtype=np.int32
            )
            landmarks_px[:, 0] = np.clip(landmarks_px[:, 0], 0, image_rgb.shape[1] - 1)
            landmarks_px[:, 1] = np.clip(landmarks_px[:, 1], 0, image_rgb.shape[0] - 1)
            
            if len(landmarks_px) >= 3:
                try:
                    hull = cv2.convexHull(landmarks_px)
                    # Make the mask slightly larger by dilating
                    cv2.fillConvexPoly(mask, hull, 255)
                    # Dilate the mask to include more hand area
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
                    mask = cv2.dilate(mask, kernel, iterations=1)
                except Exception as e:
                    logger.warning(f"Convex hull error: {e}")
                    # Fallback with larger circles
                    for pt in landmarks_px:
                        cv2.circle(mask, tuple(pt), 12, 255, -1)  # Increased radius
            else:
                for pt in landmarks_px:
                    cv2.circle(mask, tuple(pt), 12, 255, -1)  # Increased radius

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    masked_gray_image = cv2.bitwise_and(gray, gray, mask=mask)

    # Enhanced fallback strategy
    if np.any(mask > 0):  # Valid hand detected
        current_valid_mask = masked_gray_image.copy()
        return masked_gray_image, current_valid_mask, has_hands
    elif last_valid_mask is not None:  # Use last valid frame from this video
        logger.warning("No hands detected, using last valid mask")
        # Gradually fade the last valid mask to encourage new detection
        faded_mask = (last_valid_mask * 0.8).astype(np.uint8)
        return faded_mask, faded_mask, False
    else:
        logger.warning("No hands detected and no previous valid mask")
        # Try to create a basic mask based on motion/skin color as last resort
        fallback_mask = create_fallback_hand_mask(image_rgb)
        if np.any(fallback_mask > 0):
            fallback_gray = cv2.bitwise_and(gray, gray, mask=fallback_mask)
            return fallback_gray, fallback_gray, False
        return np.zeros_like(gray), None, False

def load_dependencies():
    global model_loaded, class_names_loaded, device_loaded, transform_loaded
    if model_loaded is not None:
        return

    logger.info("Server: Loading PyTorch model and dependencies...")
    device_loaded = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device_loaded}")

    class_names_path = config.CLASS_NAMES_FILE
    if not os.path.exists(class_names_path):
        raise RuntimeError(f"Class names file not found: {class_names_path}")
    with open(class_names_path, "r") as f:
        class_names_loaded = [line.strip() for line in f if line.strip()]

    model_path = config.BEST_MODEL_PATH
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model file not found: {model_path}")

    num_classes = len(class_names_loaded)
    model_loaded = SignLanguageModel(
        num_classes=num_classes,
        input_size=config.INPUT_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        dropout_rate=config.DROPOUT_RATE,
        bidirectional=config.BIDIRECTIONAL,
        num_lstm_layers=config.NUM_LSTM_LAYERS
    )

    try:
        model_loaded.load_state_dict(torch.load(model_path, map_location=device_loaded))
    except RuntimeError as e:
        if "Attempting to deserialize object on a CUDA device" in str(e) and device_loaded.type == 'cpu':
            model_loaded.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model_loaded.to(device_loaded)
    model_loaded.eval()

    normalize = transforms.Normalize(mean=[0.5], std=[0.5])
    transform_loaded = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
        transforms.ToTensor(),
        normalize
    ])

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def on_startup():
    logger.info("Server: on_startup event - Loading dependencies.")
    load_dependencies()
    get_hands_detector()
    logger.info("Server: on_startup complete.")

@app.get("/")
def root():
    return {"message": "Sign Language Translator API is running."}

@app.post("/predict_video")
async def predict_video(file: UploadFile = File(...)):
    logger.info("Starting video prediction endpoint")

    # Check dependencies
    if any(x is None for x in [model_loaded, class_names_loaded, device_loaded, transform_loaded]):
        raise HTTPException(status_code=503, detail="Model or dependencies not loaded.")
    if hands_detector_instance is None:
        raise HTTPException(status_code=503, detail="MediaPipe Hands not available.")

    request_debug_dir = os.path.join(SERVER_DEBUG_ROOT_DIR, f"video_request_{int(time.time())}")
    os.makedirs(request_debug_dir, exist_ok=True)
    temp_video_path = os.path.join(request_debug_dir, "uploaded_video.mp4")

    try:
        # Save uploaded video
        video_data = await file.read()
        with open(temp_video_path, "wb") as f:
            f.write(video_data)
        logger.info(f"Video saved to {temp_video_path}")

        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            logger.error("Could not open video file")
            raise HTTPException(status_code=400, detail="Could not open video file.")

        # Get rotation type for mobile videos
        rotation_type = get_rotation_type()

        # Get FPS and total frame count - Keep trimming to avoid camera button interference
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        trim_frames = int(0.5 * fps) if fps > 0 else 15  # Remove last 0.5s to avoid camera button interference
        valid_frame_count = max(0, total_frames - trim_frames)
        logger.info(f"Trimming last {trim_frames} frames to avoid camera button interference ({valid_frame_count} remaining)")

        # Read frames with trimming and rotation
        all_frames = []
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame_count >= valid_frame_count:  # Stop at trimmed count
                break
            if frame is None:
                continue
            
            # Apply rotation to match training data orientation
            if rotation_type is not None:
                frame = cv2.rotate(frame, rotation_type)
            
            all_frames.append(frame)
            frame_count += 1
        cap.release()
        logger.info(f"Successfully read {len(all_frames)} frames")

        if not all_frames:
            logger.error("No frames extracted from video")
            raise HTTPException(status_code=400, detail="No frames extracted from video.")

    except Exception as e:
        logger.error(f"Error during video processing: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        await file.close()

    # --- Improved Motion Detection ---
    motion_segments = []
    prev_gray = None
    motion_threshold = 8  # Slightly higher threshold
    min_segment_frames = max(config.SEQUENCE_LENGTH, 12)  # Ensure minimum frames
    stable_silence_count = 0
    current_segment = []

    for idx, frame in enumerate(all_frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None and gray.shape == prev_gray.shape:
            diff = cv2.absdiff(gray, prev_gray)
            motion = np.mean(diff)

            # Check for hands in current frame
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = get_hands_detector().process(image_rgb)
            has_hands = results.multi_hand_landmarks is not None

            if motion > motion_threshold or has_hands:
                current_segment.append(frame)
                stable_silence_count = 0
            elif current_segment:
                stable_silence_count += 1
                if stable_silence_count >= 8:  # End segment after 8 frames of low motion
                    if len(current_segment) >= min_segment_frames:
                        motion_segments.append(current_segment)
                        logger.info(f"Motion segment detected with {len(current_segment)} frames")
                    current_segment = []
                    stable_silence_count = 0
                else:
                    current_segment.append(frame)  # Include transition frames
        prev_gray = gray

    # Don't forget the last segment
    if current_segment and len(current_segment) >= min_segment_frames:
        motion_segments.append(current_segment)
        logger.info(f"Final segment detected with {len(current_segment)} frames")

    logger.info(f"Detected {len(motion_segments)} potential sign segments")

    # If no segments found, create segments from the full video
    if not motion_segments:
        logger.warning("No motion segments found. Creating segments from full video.")
        segment_size = config.SEQUENCE_LENGTH * 2  # Larger segments
        for i in range(0, len(all_frames), segment_size):
            segment = all_frames[i:i + segment_size]
            if len(segment) >= min_segment_frames:
                motion_segments.append(segment)

    # --- Predict Each Segment ---
    detected_signs = []

    for seg_idx, segment in enumerate(motion_segments):
        logger.info(f"Processing segment {seg_idx} with {len(segment)} frames")
        
        # Create separate folder for each segment
        segment_debug_dir = os.path.join(request_debug_dir, f"segment_{seg_idx:02d}")
        os.makedirs(segment_debug_dir, exist_ok=True)
        
        # Better frame selection - use temporal distribution
        if len(segment) <= config.SEQUENCE_LENGTH:
            selected_frames = segment
        else:
            # Select frames more evenly distributed across the segment
            indices = np.linspace(0, len(segment) - 1, config.SEQUENCE_LENGTH, dtype=int)
            selected_frames = [segment[i] for i in indices]
        
        processed_tensors = []
        last_valid_mask_local = None  # Local to this segment

        for frame_idx, bgr_frame in enumerate(selected_frames):
            try:
                image_rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                masked_gray, last_valid_mask_local, hands_detected = apply_mediapipe_mask_and_grayscale_internal(
                    image_rgb, last_valid_mask_local
                )

                if masked_gray is not None and np.any(masked_gray > 0):
                    # Save debug frame
                    debug_frame_path = os.path.join(segment_debug_dir, f"frame_{frame_idx:02d}_masked.png")
                    cv2.imwrite(debug_frame_path, masked_gray)

                    tensor = transform_loaded(masked_gray)
                    processed_tensors.append(tensor)
                else:
                    logger.warning(f"Empty mask for segment {seg_idx}, frame {frame_idx}")

            except Exception as e:
                logger.error(f"Error processing frame {frame_idx} in segment {seg_idx}: {e}")

        if len(processed_tensors) < config.SEQUENCE_LENGTH // 2:  # Need at least half the frames
            logger.warning(f"Insufficient valid frames for segment {seg_idx} ({len(processed_tensors)} processed)")
            continue

        # Pad or truncate to exact sequence length
        while len(processed_tensors) < config.SEQUENCE_LENGTH:
            processed_tensors.append(processed_tensors[-1])  # Repeat last frame
        processed_tensors = processed_tensors[:config.SEQUENCE_LENGTH]  # Truncate if too long

        input_tensor = torch.stack(processed_tensors).unsqueeze(0).to(device_loaded)
        logger.info(f"Input tensor shape for segment {seg_idx}: {input_tensor.shape}")

        with torch.no_grad():
            outputs = model_loaded(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, pred_idx = torch.max(probabilities, 1)

        predicted_class = class_names_loaded[pred_idx.item()]
        confidence_val = confidence.item()

        # Save segment prediction info
        segment_info_path = os.path.join(segment_debug_dir, "prediction_info.txt")
        with open(segment_info_path, "w") as f:
            f.write(f"Segment {seg_idx} Prediction Results:\n")
            f.write(f"Predicted Class: {predicted_class}\n")
            f.write(f"Confidence Score: {confidence_val:.4f}\n")
            f.write(f"Total Frames in Segment: {len(segment)}\n")
            f.write(f"Selected Frames: {len(selected_frames)}\n")
            f.write(f"Processed Tensors: {len(processed_tensors)}\n")
            f.write(f"Input Tensor Shape: {input_tensor.shape}\n")

        # Only include predictions with reasonable confidence
        if confidence_val > 0.5:  # Adjust threshold as needed
            detected_signs.append({
                "predicted_class": predicted_class,
                "confidence_score": confidence_val,
                "segment_frames": len(segment),
                "segment_id": seg_idx
            })
            logger.info(f"Segment {seg_idx}: {predicted_class} (confidence: {confidence_val:.3f})")
        else:
            logger.warning(f"Low confidence prediction ({confidence_val:.3f}) for segment {seg_idx}, skipping")

    # Save overall request summary
    summary_path = os.path.join(request_debug_dir, "request_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Video Processing Summary\n")
        f.write(f"========================\n")
        f.write(f"Total Frames: {len(all_frames)}\n")
        f.write(f"Frames Trimmed: {trim_frames}\n")
        f.write(f"Rotation Applied: {rotation_type}\n")
        f.write(f"Motion Segments Detected: {len(motion_segments)}\n")
        f.write(f"Valid Predictions: {len(detected_signs)}\n\n")
        
        f.write(f"Detected Signs:\n")
        f.write(f"---------------\n")
        for i, sign in enumerate(detected_signs):
            f.write(f"{i+1}. {sign['predicted_class']} (confidence: {sign['confidence_score']:.3f}, segment: {sign['segment_id']})\n")
        
        f.write(f"\nSegment Details:\n")
        f.write(f"----------------\n")
        for i, segment in enumerate(motion_segments):
            f.write(f"Segment {i}: {len(segment)} frames\n")

    return {
        "detected_signs": detected_signs,
        "total_segments": len(motion_segments),
        "debug_info": {
            "rotation_applied": rotation_type is not None,
            "rotation_type": str(rotation_type) if rotation_type else "None",
            "frames_trimmed": trim_frames,
            "total_frames_processed": len(all_frames)
        }
    }