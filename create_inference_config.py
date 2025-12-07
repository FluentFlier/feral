import os
import json
import argparse
from pathlib import Path

def create_config(video_dir, output_path):
    videos = [f.name for f in Path(video_dir).glob("*.mp4")]
    data = {
        "is_multilabel": False,
        "class_names": {"0": "other"},
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
    print(f"Created {output_path} with {len(videos)} videos.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_dir")
    parser.add_argument("output_path")
    args = parser.parse_args()
    create_config(args.video_dir, args.output_path)
