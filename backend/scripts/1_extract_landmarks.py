import os
import cv2
import csv
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

DATASET_PATH = r"c:\Users\divya\OneDrive\Desktop\Yoga_guide\Final_project3_dataset"
OUTPUT_CSV = r"c:\Users\divya\OneDrive\Desktop\Yoga_guide\backend\yoga_dataset.csv"

def build_csv_header():
    header = ['pose', 'quality']
    for i in range(33):
        header.extend([f'lm_{i}_x', f'lm_{i}_y', f'lm_{i}_z', f'lm_{i}_v'])
    return header

def extract_from_video(video_path, pose_name, quality, csv_writer, pose_detector):
    cap = cv2.VideoCapture(video_path)
    frames_processed = 0
    
    while cap.isOpened() and frames_processed < 50: # max 50 frames per video
        ret, frame = cap.read()
        if not ret:
            break
            
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        results = pose_detector.detect(mp_image)
        
        if results.pose_landmarks and len(results.pose_landmarks) > 0:
            row = [pose_name, quality]
            # First tracked bounding box/person
            landmarks = results.pose_landmarks[0]
            for lm in landmarks:
                row.extend([lm.x, lm.y, lm.z, getattr(lm, 'visibility', 1.0) or 1.0])
            csv_writer.writerow(row)
            frames_processed += 1
            
    cap.release()
    return frames_processed

def main():
    print("Starting Landmark Extraction for Neural Network...")
    
    model_path = 'pose_landmarker_lite.task'
    if not os.path.exists(model_path):
        url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task'
        urllib.request.urlretrieve(url, model_path)

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False)
        
    with open(OUTPUT_CSV, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(build_csv_header())
        
        with vision.PoseLandmarker.create_from_options(options) as pose_detector:
            poses = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]
            
            for pose_name in poses:
                pose_dir = os.path.join(DATASET_PATH, pose_name)
                for quality in ['good', 'avg', 'poor']:
                    quality_dir = os.path.join(pose_dir, quality)
                    if not os.path.exists(quality_dir):
                        continue
                        
                    videos = [v for v in os.listdir(quality_dir) if v.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
                    for video_file in videos:
                        video_path = os.path.join(quality_dir, video_file)
                        print(f"Processing: {pose_name} -> {quality} -> {video_file}")
                        try:
                            frames = extract_from_video(video_path, pose_name, quality, writer, pose_detector)
                            print(f"  Extracted {frames} frames.")
                        except Exception as e:
                            print(f"  Error on {video_file}: {e}")

    print(f"\nDone! Dataset saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
