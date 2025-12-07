#!/bin/bash
set -e

# Check if videos folder is empty
if [ -z "$(ls -A ../videos)" ]; then
   echo "Error: '../videos' folder is empty. Please add video files."
   exit 1
fi

# 1. Prepare Videos
echo "Preparing videos..."
python3 prepare_videos.py ../videos re_encoded

# 2. Create Inference Config
echo "Creating inference config..."
python3 create_inference_config.py re_encoded inference.json

# 3. Run Inference (if checkpoint provided)
if [ -z "$1" ]; then
    echo "---------------------------------------------------"
    echo "Setup complete!"
    echo "Videos re-encoded to: feral/re_encoded"
    echo "Inference config created: feral/inference.json"
    echo ""
    echo "To run inference, you need a trained checkpoint."
    echo "Usage: ./run_feral_pipeline.sh <path_to_checkpoint.pth>"
    echo "---------------------------------------------------"
else
    CHECKPOINT="$1"
    echo "Running inference with checkpoint $CHECKPOINT..."
    
    # Detect Device (MPS for Mac, else CPU)
    if python3 -c "import torch; print(torch.backends.mps.is_available())" | grep -q "True"; then
        DEVICE="mps"
    else
        DEVICE="cpu"
    fi
    echo "Using device: $DEVICE"
    
    # Update config device (careful with sed on Mac)
    sed -i '' "s/device: \"cuda\"/device: \"$DEVICE\"/" configs/default_vjepa.yaml
    
    # Run FERAL
    python3 run.py re_encoded inference.json --checkpoint "$CHECKPOINT"
fi
