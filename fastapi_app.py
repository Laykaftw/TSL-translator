import os
import cv2
import torch
import numpy as np
import mediapipe as mp
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms
from typing import List
import logging
import time

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

hands_detector_instance = None
model_loaded = None
class_names_loaded = None
device_loaded = None
transform_loaded = None

def get_hands_detector():
    global hands_detector_instance
    if hands_detector_instance is None:
        logger.info("Server: Initializing MediaPipe Hands (static_image_mode=False)...")
        mp_hands = mp.solutions.hands
        hands_detector_instance = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )
        logger.info("Server: MediaPipe Hands detector initialized.")
    return hands_detector_instance

def apply_mediapipe_mask_and_grayscale_internal(image_rgb: np.ndarray) -> np.ndarray:
    """Takes an RGB image, returns masked grayscale image."""
    detector = get_hands_detector()
    if detector is None:
        logger.error("Server: MediaPipe not initialized properly.")
        return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # Lock writing
    image_rgb.flags.writeable = False
    results = detector.process(image_rgb)
    image_rgb.flags.writeable = True

    mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Convert landmarks to pixel coords
            landmarks_px = np.array(
                [(lm.x * image_rgb.shape[1], lm.y * image_rgb.shape[0])
                 for lm in hand_landmarks.landmark],
                dtype=np.int32
            )
            # Clamping
            landmarks_px[:, 0] = np.clip(landmarks_px[:, 0], 0, image_rgb.shape[1] - 1)
            landmarks_px[:, 1] = np.clip(landmarks_px[:, 1], 0, image_rgb.shape[0] - 1)

            if len(landmarks_px) >= 3:
                try:
                    hull = cv2.convexHull(landmarks_px)
                    cv2.fillConvexPoly(mask, hull, 255)
                except Exception as e:
                    logger.warning(f"Server: Convex hull error: {e}")
                    for point in landmarks_px:
                        cv2.circle(mask, tuple(point), 5, (255), -1)
            else:
                for point in landmarks_px:
                    cv2.circle(mask, tuple(point), 5, (255), -1)

    # Grayscale
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    masked_gray_image = cv2.bitwise_and(gray, gray, mask=mask)
    return masked_gray_image

def load_dependencies():
    global model_loaded, class_names_loaded, device_loaded, transform_loaded
    if model_loaded is not None:
        return

    logger.info("Server: Loading model dependencies...")
    device_loaded = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Server: Using device: {device_loaded}")

    class_names_path = config.CLASS_NAMES_FILE
    if not os.path.exists(class_names_path):
        logger.error(f"Class names file not found at {class_names_path}")
        raise RuntimeError(f"Class names file not found: {class_names_path}")

    with open(class_names_path, "r") as f:
        class_names_loaded = [line.strip() for line in f if line.strip()]
    num_classes = len(class_names_loaded)
    logger.info(f"Server: Loaded {num_classes} class names.")

    model_path = config.BEST_MODEL_PATH
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        raise RuntimeError(f"Model file not found: {model_path}")

    # Initialize the model
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
        logger.warning(f"Loading model on {device_loaded}: {e}")
        if "Attempting to deserialize object on a CUDA device" in str(e) and device_loaded.type == 'cpu':
            model_loaded.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model_loaded.to(device_loaded)
    model_loaded.eval()
    logger.info("Server: Model loaded successfully.")

    # Setup transforms
    normalize = transforms.Normalize(mean=[0.5], std=[0.5])
    transform_loaded = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        normalize
    ])
    logger.info("Server: Transforms set up.")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.on_event("startup")
async def startup_event():
    logger.info("Server: Startup - loading dependencies...")
    load_dependencies()
    logger.info("Server: Dependencies loaded. Initializing MediaPipe detector...")
    get_hands_detector()
    logger.info("Server: Startup complete.")

@app.post("/predict")
async def predict_sequence(files: List[UploadFile] = File(...)):
    if model_loaded is None or class_names_loaded is None or device_loaded is None or transform_loaded is None:
        logger.error("Server: Model or transforms not loaded.")
        raise HTTPException(status_code=503, detail="Model or dependencies not loaded.")

    if hands_detector_instance is None:
        logger.error("Server: MediaPipe Hands not initialized.")
        raise HTTPException(status_code=503, detail="MediaPipe Hands not available.")

    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")
    if len(files) != config.SEQUENCE_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Please upload exactly {config.SEQUENCE_LENGTH} frames."
        )

    request_debug_dir = os.path.join(SERVER_DEBUG_ROOT_DIR, f"request_{int(time.time())}")
    os.makedirs(request_debug_dir, exist_ok=True)
    logger.info(f"Server: Debug frames saved in: {request_debug_dir}")

    processed_tensors = []
    for idx, file in enumerate(files):
        try:
            data = await file.read()
            nparr = np.frombuffer(data, np.uint8)
            bgr_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if bgr_frame is None:
                raise HTTPException(status_code=400, detail=f"Could not decode {file.filename}")

            # Save debug BGR frame
            debug_bgr_path = os.path.join(request_debug_dir, f"{idx:02d}_server_received_bgr.png")
            cv2.imwrite(debug_bgr_path, bgr_frame)

            # Convert BGR -> RGB for MediaPipe
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

            debug_rgb_path = os.path.join(request_debug_dir, f"{idx:02d}_server_converted_rgb.png")
            cv2.imwrite(debug_rgb_path, cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))  # saved for viewing

            # Apply MediaPipe Mask + Gray
            masked_gray_np = apply_mediapipe_mask_and_grayscale_internal(rgb_frame)
            debug_masked_gray_path = os.path.join(request_debug_dir, f"{idx:02d}_server_masked_gray.png")
            cv2.imwrite(debug_masked_gray_path, masked_gray_np)

            # Transform to tensor
            tensor_frame = transform_loaded(masked_gray_np)
            processed_tensors.append(tensor_frame)
        except Exception as e:
            logger.error(f"Server: Error processing {file.filename}: {e}")
            raise HTTPException(status_code=500, detail=f"Error processing {file.filename}: {e}")
        finally:
            await file.close()

    # Stack to shape (1, SEQUENCE_LENGTH, ...)
    if len(processed_tensors) != config.SEQUENCE_LENGTH:
        logger.error("Server: Not enough frames after processing.")
        raise HTTPException(status_code=500, detail="Frame processing failed or incomplete.")

    input_tensor = torch.stack(processed_tensors).unsqueeze(0).to(device_loaded)
    logger.info(f"Server: Final input_tensor shape: {tuple(input_tensor.shape)}")

    with torch.no_grad():
        outputs = model_loaded(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probabilities, 1)

    predicted_class = class_names_loaded[pred_idx.item()]
    confidence_val = confidence.item()
    logger.info(f"Server: Predicted: {predicted_class}, Confidence: {confidence_val:.4f}")

    return {
        "predicted_class": predicted_class,
        "confidence_score": confidence_val,
        "all_probabilities": probabilities.cpu().numpy().tolist()[0]
    }

@app.get("/")
def root():
    return {"message": "Sign Language Translator API is running."}