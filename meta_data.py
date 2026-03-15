import cv2
import csv
from pathlib import Path
import sys

def generate_ucf50_metadata(ucf50_folder_path):
    # 1. Resolve the absolute path
    ucf_dir = Path(ucf50_folder_path).resolve()
    
    if not ucf_dir.exists() or not ucf_dir.is_dir():
        print(f"Error: The directory '{ucf_dir}' does not exist.")
        print("Please check the path and try again.")
        sys.exit(1)

    # 2. Define the CSV output path in the parent directory
    parent_dir = ucf_dir.parent
    csv_file_path = parent_dir / "ucf50_video_metadata.csv"
    
    print(f"Scanning directory: {ucf_dir}")
    print(f"CSV will be saved to: {csv_file_path}")

    # 3. Find all .avi files
    video_files = list(ucf_dir.rglob('*.avi'))
    total_videos = len(video_files)
    
    if total_videos == 0:
        print("No .avi files found in the specified directory.")
        sys.exit(1)
        
    print(f"Found {total_videos} videos. Processing...")

    # 4. Open the CSV file for writing
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        # Added 'Duration_Sec' to the header
        writer.writerow(['Video_Path', 'FPS', 'Total_Frames', 'Duration_Sec'])

        # 5. Iterate through each video file
        for i, video_path in enumerate(video_files, 1):
            rel_path = video_path.relative_to(parent_dir)
            cap = cv2.VideoCapture(str(video_path))

            if not cap.isOpened():
                print(f"Warning: Could not open video {rel_path}. Skipping.")
                continue

            # Extract properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate duration (handle potential divide-by-zero if video is corrupted)
            if fps > 0:
                duration_sec = round(total_frames / fps, 2)
            else:
                duration_sec = 0.0

            cap.release()

            # Write the new row including duration
            writer.writerow([str(rel_path), fps, total_frames, duration_sec])

            if i % 500 == 0:
                print(f"Processed {i} / {total_videos} videos...")

    print(f"Success! Metadata extraction complete. Saved to '{csv_file_path}'")

if __name__ == "__main__":
    # Update this path if your UCF50 folder is located elsewhere
    TARGET_FOLDER = "./datasets/UCF50" 
    
    generate_ucf50_metadata(TARGET_FOLDER)
