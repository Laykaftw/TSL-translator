import cv2
import os
import time

def capture_sign_videos(sign_name, num_videos=5, duration=3):
    sign_dir = f"data/raw/{sign_name}"
    os.makedirs(sign_dir, exist_ok=True)

    # Count existing videos
    existing_videos = len([f for f in os.listdir(sign_dir) if f.endswith(".mp4")])
    print(f"Existing videos for '{sign_name}': {existing_videos}")

    # Define video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    try:
        print(f"Press SPACE to start recording {num_videos} more videos...")
        
        # Wait for SPACE key once
        while True:
            ret, frame = cap.read()
            cv2.putText(frame, "Press SPACE to start", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Recorder', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                break

        for video_num in range(1, num_videos + 1):
            video_index = existing_videos + video_num
            video_path = f"{sign_dir}/{sign_name}_{video_index:03d}.mp4"
            out = cv2.VideoWriter(video_path, fourcc, 30.0, (frame_width, frame_height))

            # 3-second countdown before each video
            for i in range(3, 0, -1):
                # print(f"Recording starts in {i}...")
                start_time = time.time()
                while time.time() - start_time < 1:
                    ret, frame = cap.read()
                    cv2.putText(frame, f"Starting in {i}...", (50, 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow('Recorder', frame)
                    cv2.waitKey(1)

            # Start recording
            # print(f"Recording video {video_num}/{num_videos + existing_videos}...")
            start_time = time.time()
            while time.time() - start_time < duration:
                ret, frame = cap.read()
                out.write(frame)
                cv2.putText(frame, f"Recording... {video_num}/{num_videos + existing_videos}", 
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Video {video_num}/{num_videos}", 
                            (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.imshow('Recorder', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                cv2.imshow('Recorder', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            out.release()
            print(f"Video saved to {video_path}")

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    sign = input("Enter the sign name: ").strip().lower()
    
    # Check existing videos
    sign_dir = f"data/raw/{sign}"
    existing_videos = len([f for f in os.listdir(sign_dir) if f.endswith(".mp4")]) if os.path.exists(sign_dir) else 0

    print(f"Existing videos for '{sign}': {existing_videos}")
    num_videos = int(input("How many more videos do you want to add? "))
    
    capture_sign_videos(sign, num_videos=num_videos, duration=3)
