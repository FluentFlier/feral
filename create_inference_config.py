import os
import json
import argparse
from pathlib import Path

def create_config(video_dir, output_path, labels_file=None):
    class_names = {"0": "other"}
    
    if labels_file and os.path.exists(labels_file):
        with open(labels_file, 'r') as f:
            lbl_data = json.load(f)
            if "class_names" in lbl_data:
                class_names = lbl_data["class_names"]
                
    videos = [f.name for f in Path(video_dir).glob("*.mp4")]
    data = {
        "is_multilabel": False,
        "class_names": class_names,
        "labels": {},
        "splits": {
            "train": [],
            "val": [],
            "test": [],
            "inference": videos
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Created {output_path} with {len(videos)} videos and classes: {list(class_names.values())}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_dir")
    parser.add_argument("output_path")
    parser.add_argument("--labels", help="Path to existing labels json to sync class names", default=None)
    args = parser.parse_args()
    create_config(args.video_dir, args.output_path, args.labels)
