"""Utilities for data loading and preprocessing with MediaPipe masking (Lazy Init)."""
import os
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import time
import mediapipe as mp # Still import mediapipe
import traceback # Import traceback for detailed error printing

# Import config here
from configs import config

# --- Global variable to hold the detector once initialized ---
hands_detector_instance = None

def get_hands_detector():
    """Initializes and returns the MediaPipe Hands detector instance."""
    global hands_detector_instance
    if hands_detector_instance is None:
        print("  [MediaPipe] Initializing Hands detector...")
        mp_hands = mp.solutions.hands
        hands_detector_instance = mp_hands.Hands(
            static_image_mode=True, # Use True for dataset processing
            max_num_hands=2,
            min_detection_confidence=0.5
        )
        print("  [MediaPipe] Hands detector initialized.")
    return hands_detector_instance

def apply_mediapipe_mask_and_grayscale(image_rgb):
    """Applies MediaPipe Hands segmentation mask and converts to grayscale."""
    # Get the detector (initializes on first call)
    detector = get_hands_detector()

    # Process the image with MediaPipe Hands
    image_rgb.flags.writeable = False
    results = detector.process(image_rgb)
    image_rgb.flags.writeable = True

    # Create a black mask
    mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)

    # Draw hand landmarks on the mask if hands are detected
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
                    # Fallback: Draw landmarks as circles if hull fails
                    for point in landmark_points:
                        cv2.circle(mask, tuple(point), 5, (255), -1)


    gray_source = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    masked_gray_image = cv2.bitwise_and(gray_source, gray_source, mask=mask)
    return masked_gray_image


class SignLanguageDataset(Dataset):
    """Dataset for sign language recognition with background removal."""
    def __init__(self, data_dir, transform=None, sequence_length=16, is_training=True):
        print(f"    [Dataset Init] Initializing with data_dir: {data_dir}") # <-- Add
        self.data_dir = data_dir
        self.transform = transform
        self.sequence_length = sequence_length
        self.is_training = is_training

        try: # <-- Add try block
            print(f"    [Dataset Init] Listing contents of {data_dir}...") # <-- Add
            if not os.path.exists(data_dir):
                raise FileNotFoundError(f"Data directory not found: {data_dir}")
            if not os.path.isdir(data_dir):
                raise NotADirectoryError(f"Path is not a directory: {data_dir}")

            dir_contents = os.listdir(data_dir)
            print(f"    [Dataset Init] Found {len(dir_contents)} items in {data_dir}.") # <-- Add

            self.classes = sorted([d for d in dir_contents
                                if os.path.isdir(os.path.join(data_dir, d))])
            print(f"    [Dataset Init] Found classes (subdirectories): {self.classes}") # <-- Add
            if not self.classes:
                print(f"    [Dataset Init] WARNING: No subdirectories found in {data_dir}. Check data path and structure.")

            self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

            self.samples = []
            self.class_counts = {}
            print(f"    [Dataset Init] Processing class directories...") # <-- Add
            for cls in self.classes:
                cls_dir = os.path.join(data_dir, cls)
                # print(f"      [Dataset Init] Processing class: {cls} in {cls_dir}") # Optional: very verbose
                # Expecting video folders inside class folders
                videos = sorted([os.path.join(cls_dir, vid) for vid in os.listdir(cls_dir)
                              if os.path.isdir(os.path.join(cls_dir, vid))])
                if not videos:
                     print(f"      [Dataset Init] WARNING: No video subdirectories found in class folder: {cls_dir}")
                self.class_counts[cls] = len(videos)
                for video_dir in videos:
                    self.samples.append((video_dir, self.class_to_idx[cls]))
            print(f"    [Dataset Init] Finished processing directories. Found {len(self.samples)} total video samples.") # <-- Add

        except FileNotFoundError as e:
            print(f"    [Dataset Init] ERROR: {e}") # <-- Add error handling
            raise # Re-raise the exception
        except NotADirectoryError as e:
            print(f"    [Dataset Init] ERROR: {e}") # <-- Add error handling
            raise # Re-raise the exception
        except PermissionError:
            print(f"    [Dataset Init] ERROR: Permission denied accessing: {data_dir} or its subdirectories.") # <-- Add error handling
            raise # Re-raise the exception
        except Exception as e:
            print(f"    [Dataset Init] ERROR during initialization: {e}") # <-- Add generic error handling
            traceback.print_exc()
            raise # Re-raise the exception

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # --- Check if samples list is populated ---
        if not self.samples:
             raise IndexError("Dataset samples list is empty. Initialization might have failed.")
        if idx >= len(self.samples):
             raise IndexError(f"Index {idx} out of bounds for {len(self.samples)} samples.")
        # --- End Check ---

        video_dir, label = self.samples[idx]

        # Look for a 'frames' subfolder first
        frames_path = os.path.join(video_dir, "frames")
        if not os.path.exists(frames_path) or not os.path.isdir(frames_path):
            # Fallback to using the video_dir itself if 'frames' doesn't exist
            frames_path = video_dir

        try:
            frame_files = sorted([f for f in os.listdir(frames_path)
                                if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        except FileNotFoundError:
             print(f"Error in __getitem__: Frames path not found: {frames_path}")
             # Return dummy data or raise error
             return torch.zeros(self.sequence_length, 1, config.INPUT_SIZE, config.INPUT_SIZE), -1 # Example dummy
        except Exception as e:
             print(f"Error listing frames in {frames_path}: {e}")
             return torch.zeros(self.sequence_length, 1, config.INPUT_SIZE, config.INPUT_SIZE), -1 # Example dummy


        if len(frame_files) == 0:
             # Check parent dir again if frames_path was the subfolder
             if frames_path != video_dir:
                  try:
                       parent_frame_files = sorted([f for f in os.listdir(video_dir)
                                                    if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
                       if parent_frame_files:
                            frame_files = parent_frame_files
                            frames_path = video_dir # Update frames_path if using parent
                       else:
                            print(f"Warning: No frame images found in {video_dir} or its 'frames' subfolder.")
                            # Return dummy data or raise error
                            return torch.zeros(self.sequence_length, 1, config.INPUT_SIZE, config.INPUT_SIZE), label # Return label if known
                  except Exception as e:
                       print(f"Error listing frames in parent {video_dir}: {e}")
                       return torch.zeros(self.sequence_length, 1, config.INPUT_SIZE, config.INPUT_SIZE), label
             else:
                  print(f"Warning: No frame images found in {video_dir}.")
                  return torch.zeros(self.sequence_length, 1, config.INPUT_SIZE, config.INPUT_SIZE), label


        # --- Frame Sampling Logic ---
        num_available_frames = len(frame_files)
        indices_to_load = []
        if num_available_frames == 0:
             # Handle case with no frames found after checks
             print(f"Error: No frames to load for {video_dir}. Returning dummy data.")
             return torch.zeros(self.sequence_length, 1, config.INPUT_SIZE, config.INPUT_SIZE), label

        if num_available_frames < self.sequence_length:
            # Repeat last frame
            indices_to_load = list(range(num_available_frames)) + [num_available_frames - 1] * (self.sequence_length - num_available_frames)
        elif num_available_frames > self.sequence_length:
            if self.is_training:
                # Random start index
                start_idx = random.randint(0, num_available_frames - self.sequence_length)
                indices_to_load = list(range(start_idx, start_idx + self.sequence_length))
            else:
                # Evenly spaced indices
                indices_to_load = np.linspace(0, num_available_frames - 1, self.sequence_length).astype(int)
        else: # Exactly sequence_length frames
            indices_to_load = list(range(self.sequence_length))
        # --- End Frame Sampling ---

        frames = []
        for i in indices_to_load:
            frame_file = frame_files[i]
            frame_path = os.path.join(frames_path, frame_file)
            frame = cv2.imread(frame_path)

            if frame is None:
                print(f"Warning: Error loading frame {frame_path}. Using blank gray frame.")
                # Create a blank GRAY frame as fallback, matching expected input size
                processed_frame = np.zeros((config.INPUT_SIZE, config.INPUT_SIZE), dtype=np.uint8)
            else:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # --- Apply MediaPipe Mask and Grayscale (uses lazy init now) ---
                try:
                    processed_frame = apply_mediapipe_mask_and_grayscale(frame_rgb)
                except Exception as e:
                    print(f"Error applying MediaPipe to {frame_path}: {e}. Using blank gray frame.")
                    # Fallback to blank gray frame matching input size
                    processed_frame = np.zeros((config.INPUT_SIZE, config.INPUT_SIZE), dtype=np.uint8)
                # --- End Apply ---

            # Apply other transforms (Resize, Augment, ToTensor, Normalize)
            transformed_frame = None
            if self.transform:
                # Pass the single-channel grayscale image to the transform pipeline
                try:
                    transformed_frame = self.transform(processed_frame)
                except Exception as e:
                    print(f"Error applying transforms to frame from {frame_path}: {e}")
                    # Fallback to zero tensor
                    transformed_frame = torch.zeros(1, config.INPUT_SIZE, config.INPUT_SIZE)


            # Ensure output tensor has 1 channel, correct size
            final_frame = transformed_frame if transformed_frame is not None else torch.zeros(1, config.INPUT_SIZE, config.INPUT_SIZE)

            # Validate shape after transform
            if len(final_frame.shape) == 2: # If ToTensor didn't add channel dim
                final_frame = final_frame.unsqueeze(0)
            elif len(final_frame.shape) == 3 and final_frame.shape[0] != 1: # If channel dim is wrong
                print(f"Warning: Unexpected channel dimension {final_frame.shape[0]} after transform for {frame_path}. Taking first channel.")
                final_frame = final_frame[0, :, :].unsqueeze(0)

            # Ensure correct spatial size (Resize should be in transform, but double-check)
            if final_frame.shape[1] != config.INPUT_SIZE or final_frame.shape[2] != config.INPUT_SIZE:
                # Apply resize if not done correctly in transform (less ideal but fallback)
                print(f"Warning: Frame size mismatch ({final_frame.shape}) after transform for {frame_path}. Resizing again.")
                resize_op = transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE), antialias=True) # Add antialias
                final_frame = resize_op(final_frame)

            # Final check for 1 channel
            if final_frame.shape[0] != 1:
                print(f"Error: Final frame does not have 1 channel after all checks: {final_frame.shape} for {frame_path}. Using zero tensor.")
                # Fallback to zero tensor
                final_frame = torch.zeros(1, config.INPUT_SIZE, config.INPUT_SIZE)


            frames.append(final_frame)

        # Ensure we have the correct number of frames before stacking
        if len(frames) != self.sequence_length:
            print(f"Error: Incorrect number of frames ({len(frames)}) collected for sequence {idx}. Expected {self.sequence_length}. Padding/Truncating.")
            # Pad or truncate if necessary (should ideally not happen with sampling logic)
            if len(frames) < self.sequence_length:
                if frames: frames.extend([frames[-1]] * (self.sequence_length - len(frames)))
                else: frames = [torch.zeros(1, config.INPUT_SIZE, config.INPUT_SIZE)] * self.sequence_length
            else:
                frames = frames[:self.sequence_length]


        sequence = torch.stack(frames) # Shape: (seq_len, 1, H, W)
        return sequence, label


def get_data_loaders(data_dir, batch_size=16, sequence_length=16, input_size=128,
                    shuffle=True, num_workers=2, validation_split=0.2):
    """Create train and validation data loaders for grayscale masked data."""
    normalize = transforms.Normalize(mean=[0.5], std=[0.5])
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((input_size + 10, input_size + 10), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0), antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        transforms.ToTensor(),
        normalize
    ])
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.ToTensor(),
        normalize
    ])

    print("  [DataLoader] Initializing SignLanguageDataset...") # <-- Add
    try:
        dataset = SignLanguageDataset(
            data_dir=data_dir,
            transform=train_transform,
            sequence_length=sequence_length,
            is_training=True
        )
    except (FileNotFoundError, NotADirectoryError, PermissionError) as e:
         print(f"  [DataLoader] CRITICAL ERROR: Failed to initialize dataset: {e}")
         return None, None, [] # Return empty values if dataset init fails
    except Exception as e:
         print(f"  [DataLoader] CRITICAL ERROR: Unexpected error initializing dataset: {e}")
         traceback.print_exc()
         return None, None, [] # Return empty values

    print("  [DataLoader] SignLanguageDataset initialized.") # <-- Add

    # --- Check if dataset loading actually found samples ---
    if len(dataset) == 0:
         print("  [DataLoader] CRITICAL ERROR: Dataset initialization succeeded but found 0 samples. Check data directory structure and contents.")
         return None, None, []
    if not dataset.classes:
         print("  [DataLoader] CRITICAL ERROR: Dataset initialization succeeded but found 0 classes. Check data directory structure.")
         return None, None, []
    # --- End Check ---


    print("Dataset loaded (Grayscale & Masking - Lazy Init):")
    print(f"- Total samples: {len(dataset)}")
    print(f"- Classes: {', '.join(dataset.classes)}")
    print("- Class distribution:")
    for cls, count in dataset.class_counts.items(): print(f"  - {cls}: {count} samples")

    # Split indices
    try:
        # Ensure stratification is possible
        labels_for_split = [dataset.samples[i][1] for i in range(len(dataset))]
        unique_labels, counts = np.unique(labels_for_split, return_counts=True)
        min_samples_per_class = counts.min()
        n_splits_required = max(2, int(1 / validation_split)) # Approx splits needed for test_size

        if min_samples_per_class < n_splits_required:
             print(f"Warning: The least populated class ({dataset.classes[unique_labels[counts.argmin()]]}) has only {min_samples_per_class} samples, which is less than the number required for stratified splits ({n_splits_required}). Using non-stratified split.")
             raise ValueError("Not enough samples in minority class for stratification.")

        train_indices, val_indices = train_test_split(
            list(range(len(dataset))),
            test_size=validation_split,
            stratify=labels_for_split,
            random_state=42
        )
    except ValueError as e:
         print(f"Warning: Stratified split failed ({e}). Using non-stratified split.")
         num_samples = len(dataset); indices = list(range(num_samples))
         split = int(np.floor(validation_split * num_samples))
         np.random.seed(42); np.random.shuffle(indices)
         train_indices, val_indices = indices[split:], indices[:split]

    # Create samplers
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

    # Create data loaders
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        # Add persistent_workers=True if num_workers > 0 and PyTorch version supports it
        # persistent_workers=True if num_workers > 0 else False,
    )

    # Create a separate dataset instance for validation with val_transform
    try:
        val_dataset = SignLanguageDataset(
            data_dir=data_dir,
            transform=val_transform, # Use validation transform
            sequence_length=sequence_length,
            is_training=False
        )
    except (FileNotFoundError, NotADirectoryError, PermissionError) as e:
         print(f"  [DataLoader] CRITICAL ERROR: Failed to initialize validation dataset: {e}")
         return None, None, [] # Return empty values if dataset init fails
    except Exception as e:
         print(f"  [DataLoader] CRITICAL ERROR: Unexpected error initializing validation dataset: {e}")
         traceback.print_exc()
         return None, None, [] # Return empty values

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size, # Can use a larger batch size for validation if memory allows
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        # persistent_workers=True if num_workers > 0 else False,
    )

    print(f"Created data loaders: {len(train_indices)} training, {len(val_indices)} validation")

    return train_loader, val_loader, dataset.classes


if __name__ == '__main__':
    print("Testing dataset loading with lazy masking...")
    data_dir = config.PROCESSED_DATA_DIR
    if not os.path.exists(data_dir): print(f"Error: Data directory '{data_dir}' not found.")
    else:
        try:
            # Use num_workers=0 for easier debugging in __main__
            test_loader, _, classes = get_data_loaders(data_dir, batch_size=4, sequence_length=config.SEQUENCE_LENGTH, input_size=config.INPUT_SIZE, num_workers=0)
            if test_loader:
                print("\nFetching one batch...")
                start_fetch = time.time()
                sequences, labels = next(iter(test_loader)) # This will trigger MediaPipe init and __getitem__
                fetch_time = time.time() - start_fetch
                print(f"Batch fetched successfully in {fetch_time:.2f}s!")
                print("Sequence shape:", sequences.shape) # Should be (batch, seq_len, 1, H, W)
                print("Labels:", labels)
                print("Class mapping:", {i: name for i, name in enumerate(classes)})

                # Visualize the first frame of the first sequence in the batch
                first_frame_tensor = sequences[0, 0, 0, :, :] # Batch 0, Seq 0, Channel 0
                # Denormalize (mean=0.5, std=0.5) -> (val * 0.5) + 0.5
                first_frame_np = first_frame_tensor.numpy() * 0.5 + 0.5
                first_frame_np = np.clip(first_frame_np * 255, 0, 255).astype(np.uint8)

                cv2.imshow("Sample Masked Grayscale Frame (Normalized)", first_frame_np)
                print("Displaying sample frame. Press any key to close.")
                cv2.waitKey(0); cv2.destroyAllWindows()
            else: print("Failed to create data loader.")
        except Exception as e: print(f"Error during dataset test: {e}"); traceback.print_exc()
    # Note: No explicit detector.close() needed here as it's managed within the function scope now