import subprocess
import os
import json
import threading

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_SCRIPT = os.path.join(BASE_DIR, "run.py")
PREPARE_VIDEOS_SCRIPT = os.path.join(BASE_DIR, "prepare_videos.py")
INFERENCE_CONFIG_SCRIPT = os.path.join(BASE_DIR, "create_inference_config.py")

def run_training_background(video_folder, label_file):
    """
    Runs the training script in a background thread/process.
    For this demo, we'll just launch it efficiently.
    """
    cmd = ["python3", TRAIN_SCRIPT, video_folder, label_file]
    
    # We can use Popen to let it run detached
    log_file = open(os.path.join(BASE_DIR, "dashboard_train.log"), "w")
    process = subprocess.Popen(cmd, cwd=BASE_DIR, stdout=log_file, stderr=subprocess.STDOUT)
    return process.pid

def run_inference(raw_video_path):
    """
    Runs the full inference pipeline for a single video.
    1. Re-encode video
    2. Create config
    3. Run inference
    """
    video_name = os.path.basename(raw_video_path)
    # Temporary inference folder
    inf_dir = os.path.join(BASE_DIR, "temp_inference_videos")
    re_encoded_dir = os.path.join(BASE_DIR, "temp_re_encoded")
    os.makedirs(inf_dir, exist_ok=True)
    os.makedirs(re_encoded_dir, exist_ok=True)
    
    # Symlink/Copy video to temp dir for processing
    dest = os.path.join(inf_dir, video_name)
    if not os.path.exists(dest):
        os.symlink(raw_video_path, dest)
        
    # 1. Re-encode
    subprocess.run(["python3", PREPARE_VIDEOS_SCRIPT, inf_dir, re_encoded_dir], cwd=BASE_DIR, check=True)
    
    # 2. Config
    config_path = os.path.join(BASE_DIR, "temp_inference.json")
    subprocess.run(["python3", INFERENCE_CONFIG_SCRIPT, re_encoded_dir, config_path], cwd=BASE_DIR, check=True)
    
    # 3. Find Checkpoint (take the latest best)
    checkpoints = sorted(
        [os.path.join(BASE_DIR, "checkpoints", f) for f in os.listdir(os.path.join(BASE_DIR, "checkpoints")) if f.endswith(".pt")],
        key=os.path.getmtime,
        reverse=True
    )
    if not checkpoints:
        return "No checkpoint found!"
        
    best_checkpoint = checkpoints[0]
    
    # 4. Run Inference
    # We need to handle the interactive WandB prompt or run in a mode that suppresses it.
    # The user manual says: "Do you want logs..." -> "open"
    # We can try to pipe "open" or ensure wandb is configured.
    # For now, let's assume 'open' works if piped.
    
    cmd = ["python3", TRAIN_SCRIPT, re_encoded_dir, config_path, "--checkpoint", best_checkpoint]
    
    try:
        # Run and capture output
        res = subprocess.run(cmd, cwd=BASE_DIR, input="open\n", text=True, capture_output=True)
        if res.returncode != 0:
            return f"Error running inference: {res.stderr}"
        
        # Parse output to find where result is saved?
        # FERAL saves to answers/_inference_...
        # We can just look for the newest file in answers/
        answers_dir = os.path.join(BASE_DIR, "answers")
        answers = sorted(
            [os.path.join(answers_dir, f) for f in os.listdir(answers_dir) if f.startswith("_inference")],
            key=os.path.getmtime,
            reverse=True
        )
        if answers:
            return f"Inference Complete! Results saved to: {os.path.basename(answers[0])}"
        else:
            return "Inference ran but no output file found."
            
    except Exception as e:
        return f"Exception during inference: {str(e)}"
