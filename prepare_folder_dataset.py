import os
import json
import argparse
import random
import cv2
from pathlib import Path

VIDEO_EXTS = {
    ".mts", ".m2ts", ".mp4", ".mov", ".m4v", ".avi", ".mkv", ".wmv",
    ".mpg", ".mpeg", ".mp2", ".mpv", ".3gp", ".3g2", ".webm",
}

def get_frame_count(path):
    cap = cv2.VideoCapture(str(path))
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return n if n > 0 else 0
    finally:
        cap.release()

def process_folder_structure(root_path, output_json, train_frac=0.7, val_frac=0.15):
    root = Path(root_path)
    if not root.exists():
        print(f"❌ Error: {root} does not exist.")
        return

    # 1. Identify Behaviors (Folders)
    classes = [d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")]
    if not classes:
        print(f"❌ No subdirectories found in {root}. Create folders named after behaviors.")
        return
    
    classes.sort()
    print(f"Found {len(classes)} behaviors: {classes}")
    
    # Map class names to IDs (1-based index)
    class_to_id = {name: i + 1 for i, name in enumerate(classes)}
    
    labels = {}
    all_videos = []

    # 2. Scan Videos and Assign Labels
    for class_name in classes:
        class_dir = root / class_name
        cid = class_to_id[class_name]
        
        videos = [f for f in class_dir.iterdir() if f.suffix.lower() in VIDEO_EXTS]
        print(f"Processing '{class_name}': {len(videos)} videos")
        
        for video_file in videos:
            # Get relative path for key (e.g., "walking/clip1.mp4")
            # NOTE: FERAL expects relative path from the video root
            rel_path = f"{class_name}/{video_file.name}"
            
            # Get frame count
            n_frames = get_frame_count(video_file)
            if n_frames == 0:
                print(f"⚠️  Skipping empty/unreadable video: {video_file.name}")
                continue
                
            # Create label list: [class_id, class_id, ...] for every frame
            labels[rel_path] = [cid] * n_frames
            all_videos.append(rel_path)

    if not all_videos:
        print("❌ No valid videos found.")
        return

    # 3. Create Splits
    random.shuffle(all_videos)
    n = len(all_videos)
    n_train = max(1, int(n * train_frac))
    n_val = int(n * val_frac)
    
    splits = {
        "train": all_videos[:n_train],
        "val": all_videos[n_train:n_train + n_val],
        "test": all_videos[n_train + n_val:],
        "inference": list(set(all_videos)) # Use set to ensure uniqueness if needed, but list is fine
    }

    # 4. Save JSON
    output_data = {
        "is_multilabel": False,
        "class_names": {str(cid): name for name, cid in class_to_id.items()},
        "labels": labels,
        "splits": splits
    }
    
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, separators=(",", ":"), ensure_ascii=False)
        
    print(f"✅ Success! Saved labels to {output_json}")
    print(f"  - Total videos: {n}")
    print(f"  - Classes: {len(classes)}")
    print(f"  - Videos per split: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert folder structure to FERAL labels")
    parser.add_argument("root_path", help="Path to training data root (containing behavior folders)")
    parser.add_argument("output_json", help="Path to output JSON file")
    
    args = parser.parse_args()
    process_folder_structure(args.root_path, args.output_json)
