import os
import cv2
import numpy as np
from tqdm import tqdm

# Direct paths
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"

def extract_frames(video_path, output_dir, target_fps=10):
    """Extract frames from a video file.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save frames
        target_fps: Target frames per second to extract
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return False
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame interval to achieve target_fps
    frame_interval = max(1, int(fps / target_fps))
    
    # Extract frames
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save every nth frame
        if frame_count % frame_interval == 0:
            # Save frame
            frame_path = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_count += 1
        
        frame_count += 1
    
    # Release resources
    cap.release()
    
    print(f"Extracted {saved_count} frames from {video_path}")
    return True

def preprocess_data():
    """Process raw videos into frames for model training."""
    raw_dir = RAW_DATA_DIR
    output_dir = PROCESSED_DATA_DIR
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all class directories
    class_dirs = [d for d in os.listdir(raw_dir) 
                 if os.path.isdir(os.path.join(raw_dir, d))]
    
    for class_name in class_dirs:
        print(f"\nProcessing class: {class_name}")
        
        # Create class directory in output
        class_out_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_out_dir, exist_ok=True)
        
        # Find all videos for this class
        class_dir = os.path.join(raw_dir, class_name)
        video_files = [f for f in os.listdir(class_dir) 
                      if f.endswith(('.mp4', '.avi', '.mov'))]
        
        # Process each video
        for i, video_file in enumerate(tqdm(video_files, desc=f"Processing {class_name}")):
            video_path = os.path.join(class_dir, video_file)
            
            # Create an output directory for this video
            video_name = os.path.splitext(video_file)[0]
            video_out_dir = os.path.join(class_out_dir, f"video_{i:03d}")
            frames_dir = os.path.join(video_out_dir, "frames")
            
            # Extract frames
            extract_frames(video_path, frames_dir)

if __name__ == "__main__":
    print(f"Starting preprocessing from {RAW_DATA_DIR} to {PROCESSED_DATA_DIR}")
    preprocess_data()
    print("Preprocessing complete!")