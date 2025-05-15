import requests
import cv2
import numpy as np
import os
import time
import shutil

try:
    from configs import config
except ImportError:
    class ConfigFallback:
        SEQUENCE_LENGTH = 16
    config = ConfigFallback()

def main():
    api_url = "http://localhost:8000/predict"
    client_temp_dir = "client_temp_processing_files"
    raw_frames_subdir = os.path.join(client_temp_dir, "raw_bgr_frames_for_upload")
    temp_video_filename = os.path.join(client_temp_dir, "temp_capture.mp4")

    # This directory holds raw frames from the captured video
    client_debug_raw_frames_dir = os.path.join(client_temp_dir, "debug_client_raw_video_frames")
    
    # Clean and recreate necessary folders
    os.makedirs(client_temp_dir, exist_ok=True)
    os.makedirs(raw_frames_subdir, exist_ok=True)
    os.makedirs(client_debug_raw_frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        shutil.rmtree(client_temp_dir, ignore_errors=True)
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps < 1:
        fps = 20.0 
    print(f"Webcam properties: {frame_width}x{frame_height} @ {fps:.2f} FPS (reported/defaulted)")

    print("Showing webcam feed.")
    print("Press 'SPACE' to start 3s countdown for video capture.")
    print("Press 'q' to quit.")

    video_record_duration = 3
    countdown_duration = 3  

    app_state = "idle"
    countdown_start_time = 0
    video_writer = None
    recording_start_time = 0
    recorded_video_successfully = False

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            print("Error: Can't receive frame. Exiting ...")
            app_state = "finished"
            break

        display_frame = frame_bgr.copy()
        current_time = time.time()

        if app_state == "idle":
            cv2.putText(display_frame, "Press SPACE to start", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif app_state == "countdown":
            remaining_time = countdown_duration - (current_time - countdown_start_time)
            if remaining_time > 0:
                cv2.putText(display_frame, f"Starting in: {int(np.ceil(remaining_time))}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                app_state = "recording_video"
                print("Countdown finished. Starting video recording...")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(temp_video_filename, fourcc, float(fps), (frame_width, frame_height))
                recording_start_time = current_time
        elif app_state == "recording_video":
            if video_writer:
                video_writer.write(frame_bgr)
                elapsed_recording_time = current_time - recording_start_time
                cv2.putText(display_frame, f"Recording: {elapsed_recording_time:.1f}s / {video_record_duration}s",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if elapsed_recording_time >= video_record_duration:
                    print("Video recording finished.")
                    recorded_video_successfully = True
                    break
            else:
                print("Error: VideoWriter not initialized.")
                app_state = "finished"
                break

        cv2.imshow('Webcam - Client', display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("Quitting...")
            app_state = "finished"
            break
        elif key == ord(' ') and app_state == "idle":
            print("Spacebar pressed. Starting countdown...")
            app_state = "countdown"
            countdown_start_time = current_time

    if video_writer:
        video_writer.release()
    cap.release()
    cv2.destroyAllWindows()

    # If the recording wasn't completed successfully, clean up files
    if not recorded_video_successfully:
        if app_state != "finished":
            print("Exited before completing video recording or error occurred.")
        shutil.rmtree(client_temp_dir, ignore_errors=True)
        return

    print(f"Client: Extracting frames from recorded video: {temp_video_filename}")
    video_cap = cv2.VideoCapture(temp_video_filename)
    if not video_cap.isOpened():
        print(f"Error: Could not open recorded video file: {temp_video_filename}")
        shutil.rmtree(client_temp_dir, ignore_errors=True)
        return

    all_frames_bgr = []
    while True:
        ret_vid, frame_vid_bgr = video_cap.read()
        if not ret_vid:
            break
        all_frames_bgr.append(frame_vid_bgr)
    video_cap.release()

    if not all_frames_bgr:
        print("Error: No frames extracted from the recorded video.")
        shutil.rmtree(client_temp_dir, ignore_errors=True)
        return

    num_total_frames = len(all_frames_bgr)
    print(f"Extracted {num_total_frames} total BGR frames from the video.")

    # Select frames
    if num_total_frames >= config.SEQUENCE_LENGTH:
        indices = np.linspace(0, num_total_frames - 1, config.SEQUENCE_LENGTH, dtype=int)
        selected_frames_bgr = [all_frames_bgr[i] for i in indices]
    else:
        print(f"Warning: Fewer frames ({num_total_frames}) than SEQUENCE_LENGTH ({config.SEQUENCE_LENGTH}). Using all available frames.")
        selected_frames_bgr = all_frames_bgr

    print(f"Client: Preparing {len(selected_frames_bgr)} raw BGR frames for sending...")
    bgr_frame_paths_for_upload = []
    for i, bgr_frame in enumerate(selected_frames_bgr):
        try:
            # Save raw BGR frame for local debugging
            raw_debug_filename = os.path.join(client_debug_raw_frames_dir, f"client_raw_bgr_frame_{i:02d}.png")
            cv2.imwrite(raw_debug_filename, bgr_frame)

            # Save a copy for uploading
            upload_filename = os.path.join(raw_frames_subdir, f"raw_bgr_frame_for_upload_{i:02d}.png")
            cv2.imwrite(upload_filename, bgr_frame)
            bgr_frame_paths_for_upload.append(upload_filename)
        except Exception as e:
            print(f"Client: Skipped frame {i} due to error: {e}")
            continue

    if len(bgr_frame_paths_for_upload) < min(config.SEQUENCE_LENGTH, num_total_frames):
        print(f"Client Error: Got only {len(bgr_frame_paths_for_upload)} frames to upload.")
        shutil.rmtree(client_temp_dir, ignore_errors=True)
        return

    # Prepare files for request
    files_for_request = []
    for path_to_frame in bgr_frame_paths_for_upload:
        files_for_request.append(
            ('files', (os.path.basename(path_to_frame), open(path_to_frame, 'rb'), 'image/png'))
        )

    try:
        print(f"Client: Sending {len(files_for_request)} raw BGR frames to {api_url}...")
        start_t = time.time()
        response = requests.post(api_url, files=files_for_request)
        elapsed_t = time.time() - start_t
        print(f"Client: Request took {elapsed_t:.2f} seconds.")

        if response.status_code == 200:
            print("Client: Prediction successful!")
            print("Response JSON:", response.json())
        else:
            print(f"Client: Error - Status Code: {response.status_code}")
            try:
                print("Response JSON:", response.json())
            except:
                print("Response Text:", response.text)
    except Exception as e:
        print(f"Client: Error while sending frames: {e}")
    finally:
        for _, file_tuple_item in files_for_request:
            if len(file_tuple_item) == 3 and hasattr(file_tuple_item[1], 'close'):
                file_tuple_item[1].close()
        print(f"Raw frames, debug frames, and other temp files in: {client_temp_dir}")
        print("You may want to clean them up manually after troubleshooting.")

if __name__ == "__main__":
    main()