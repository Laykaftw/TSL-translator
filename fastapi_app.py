# fastapi_server.py
import os
import cv2
import torch
import numpy as np
import mediapipe as mp
import time
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from torchvision import transforms

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
last_valid_mask = None  # Store last valid mask

def get_hands_detector():
    global hands_detector_instance
    if hands_detector_instance is None:
        logger.info("Server: Initializing MediaPipe Hands...")
        mp_hands = mp.solutions.hands
        hands_detector_instance = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.2,
        )
        logger.info("Server: MediaPipe Hands detector initialized.")
    return hands_detector_instance


def apply_mediapipe_mask_and_grayscale_internal(image_rgb: np.ndarray) -> np.ndarray:
    """Apply MediaPipe masking + grayscale conversion."""
    detector = get_hands_detector()
    image_rgb.flags.writeable = False
    results = detector.process(image_rgb)
    image_rgb.flags.writeable = True

    mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    if results.multi_hand_landmarks:
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
                    cv2.fillConvexPoly(mask, hull, 255)
                except Exception as e:
                    logger.warning(f"Convex hull error: {e}")
                    for pt in landmarks_px:
                        cv2.circle(mask, tuple(pt), 5, 255, -1)
            else:
                for pt in landmarks_px:
                    cv2.circle(mask, tuple(pt), 5, 255, -1)

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    masked_gray_image = cv2.bitwise_and(gray, gray, mask=mask)

    global last_valid_mask
    if np.any(mask > 0):  # Valid hand detected
        last_valid_mask = masked_gray_image.copy()
    elif last_valid_mask is not None:  # Use last valid frame
        masked_gray_image = last_valid_mask.copy()
    else:
        masked_gray_image = np.zeros_like(gray)  # Final fallback

    return masked_gray_image


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

        # Get FPS and total frame count
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        trim_frames = int(0.5 * fps)
        valid_frame_count = max(0, total_frames - trim_frames)
        logger.info(f"Trimming last {trim_frames} frames ({valid_frame_count} remaining)")

        # Read frames
        all_frames = []
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame_count >= valid_frame_count:
                break
            if frame is None:
                logger.warning(f"Received None frame at count {frame_count}")
                continue
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

    # --- Motion Detection ---
    motion_segments = []
    prev_gray = None
    motion_threshold = 5
    min_segment_frames = config.SEQUENCE_LENGTH // 2
    stable_silence_count = 0
    current_segment = []

    mp_hands = mp.solutions.hands
    detector = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.4
    )

    for idx, frame in enumerate(all_frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None and gray.shape == prev_gray.shape:
            diff = cv2.absdiff(gray, prev_gray)
            motion = np.mean(diff)

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.process(image_rgb)
            has_hands = results.multi_hand_landmarks is not None

            if motion > motion_threshold or has_hands:
                current_segment.append(frame)
                stable_silence_count = 0
            elif current_segment:
                stable_silence_count += 1
                if stable_silence_count >= 10:
                    if len(current_segment) >= min_segment_frames:
                        motion_segments.append(current_segment)
                        logger.info(f"Segment detected with {len(current_segment)} frames")
                    else:
                        logger.warning(f"Discarded short segment ({len(current_segment)} frames)")
                    current_segment = []
                    stable_silence_count = 0
        prev_gray = gray

    if current_segment and len(current_segment) >= min_segment_frames:
        motion_segments.append(current_segment)
        logger.info(f"Final segment detected with {len(current_segment)} frames")

    logger.info(f"Detected {len(motion_segments)} potential sign segments")

    if not motion_segments:
        logger.warning("No motion segments found. Using full video.")
        motion_segments = [all_frames]

    # --- Predict Each Segment ---
    detected_signs = []

    for seg_idx, segment in enumerate(motion_segments):
        logger.info(f"Processing segment {seg_idx} with {len(segment)} frames")
        indices = np.linspace(0, len(segment) - 1, config.SEQUENCE_LENGTH, dtype=int)
        selected_frames = [segment[i] for i in indices]
        processed_tensors = []

        for frame_idx, bgr_frame in enumerate(selected_frames):
            try:
                logger.info(f"Processing frame {frame_idx}: {bgr_frame.shape}")
                # Rotate frame         
                bgr_frame = cv2.rotate(bgr_frame, cv2.ROTATE_90_CLOCKWISE)
                logger.info(f"Frame rotated to landscape mode")

                # Convert to RGB
                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                logger.info(f"Converted to RGB")

                # Apply MediaPipe mask and grayscale
                masked_gray = apply_mediapipe_mask_and_grayscale_internal(rgb_frame)
                logger.info(f"Applied MediaPipe mask and grayscale")

                # Save debug frame
                cv2.imwrite(os.path.join(request_debug_dir, f"seg{seg_idx}_frame_{frame_idx:02d}_rotated.jpg"), bgr_frame)
                cv2.imwrite(os.path.join(request_debug_dir, f"seg{seg_idx}_frame_{frame_idx:02d}_masked.jpg"), masked_gray)

                # Apply final transform
                frame_tensor = transform_loaded(masked_gray)
                logger.info(f"Applied transform. Tensor shape: {frame_tensor.shape}")

                processed_tensors.append(frame_tensor)

            except Exception as e:
                logger.error(f"Error processing frame {frame_idx}: {e}", exc_info=True)
                continue

        if not processed_tensors:
            logger.warning(f"No tensors processed for segment {seg_idx}")
            continue

        input_tensor = torch.stack(processed_tensors).unsqueeze(0).to(device_loaded)
        logger.info(f"Input tensor shape: {input_tensor.shape}")

        with torch.no_grad():
            outputs = model_loaded(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, pred_idx = torch.max(probabilities, 1)

        predicted_class = class_names_loaded[pred_idx.item()]
        confidence_val = confidence.item()

        detected_signs.append({
            "predicted_class": predicted_class,
            "confidence_score": confidence_val
        })

    return {
        "detected_signs": detected_signs
    }