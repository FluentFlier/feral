#!/bin/bash
set -e

# Check if labels folder is empty
if [ -z "$(ls -A ../labels)" ]; then
   echo "Error: '../labels' folder is empty. Please add BORIS annotation files (.tsv or .csv)."
   exit 1
fi

# 1. Prepare Labels
echo "Converting BORIS labels..."
python3 prepare_labels.py ../labels --mode single

# 2. Run Training
echo "Starting training..."
# Detect Device
if python3 -c "import torch; print(torch.backends.mps.is_available())" | grep -q "True"; then
    DEVICE="mps"
else
    DEVICE="cpu"
fi
echo "Using device: $DEVICE"

# Update config device
sed -i '' "s/device: \"cuda\"/device: \"$DEVICE\"/" configs/default_vjepa.yaml

# Run FERAL Training
python3 run.py re_encoded ../labels/feral_behavioral_labels.json
