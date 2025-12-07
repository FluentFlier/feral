import streamlit as st
import os
import glob
import json
import pandas as pd
import backend
from backend import save_annotations_to_disk

# Config
DATA_DIR = "../"
# Use re-encoded videos for browser compatibility
RAW_VIDEOS_DIR = os.path.join(DATA_DIR, "re_encoded_inference") 
TRAINING_DIR = os.path.join(DATA_DIR, "training_data")
LABELS_FILE = os.path.join(DATA_DIR, "feral_behavioral_labels.json")

if "annotations" not in st.session_state:
    st.session_state.annotations = {}
if "video_lengths" not in st.session_state:
    st.session_state.video_lengths = {}

def load_annotations():
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, "r") as f:
            data = json.load(f)
            if "class_names" in data:
                st.session_state.class_names = data["class_names"]

def save_annotation(video_name, start_frame, end_frame, label_id):
    if video_name not in st.session_state.annotations:
        st.session_state.annotations[video_name] = []
    
    st.session_state.annotations[video_name].append({
        "start": start_frame,
        "end": end_frame,
        "label": label_id
    })
    st.success(f"Added annotation for {video_name}: Class {label_id} ({start_frame}-{end_frame})")

st.set_page_config(page_title="FERAL Dashboard", layout="wide")
st.title("FERAL: Behavioral Analysis Dashboard")

with st.sidebar:
    st.header("Navigation")
    page = st.radio("Go to", ["Annotation", "Training", "Inference"])
    
    st.divider()
    st.subheader("System Status")
    # Check for logs
    if os.path.exists("dashboard_train.log"):
        st.info("Training log found.")
    else:
        st.write("No active training log.")

if page == "Annotation":
    st.header("Video Annotation")
    video_files = glob.glob(os.path.join(RAW_VIDEOS_DIR, "*.mp4"))
    if not video_files:
        st.warning("No videos found in raw_videos.")
    else:
        video_names = [os.path.basename(v) for v in video_files]
        selected_video_name = st.selectbox("Select Video", video_names)
        selected_video_path = os.path.join(RAW_VIDEOS_DIR, selected_video_name)
        
        # Read as bytes to ensure access
        if os.path.exists(selected_video_path):
            video_bytes = open(selected_video_path, 'rb').read()
            st.video(video_bytes)
        else:
            st.error(f"File not found: {selected_video_path}")
        
        # Get video duration/frames for slider
        try:
            from decord import VideoReader, cpu
            vr = VideoReader(selected_video_path, ctx=cpu(0))
            total_frames = len(vr)
        except Exception as e:
            st.error(f"Could not read video metadata: {e}")
            total_frames = 1000 # Fallback

        st.subheader("Add Annotation")
        # Range slider
        segment_range = st.slider(
            "Select Frame Range",
            min_value=0,
            max_value=total_frames,
            value=(0, min(100, total_frames)),
            step=1
        )
        
        # Save length to session
        st.session_state.video_lengths[selected_video_name] = total_frames
        
        start_frame, end_frame = segment_range
        st.caption(f"Selected Segment: Frames {start_frame} to {end_frame}")

        # Quick Class Buttons
        st.subheader("Class Selection")
        col1, col2, col3 = st.columns(3)
        label_id = 0
        with col1:
             if st.button("😴 Sleep (1)"):
                 save_annotation(selected_video_name, start_frame, end_frame, 1)
        with col2:
             if st.button("📡 Antenna (2)"):
                 save_annotation(selected_video_name, start_frame, end_frame, 2)
        with col3:
             label_id = st.number_input("Custom ID", min_value=0, value=1)
             if st.button("Add Custom"):
                  save_annotation(selected_video_name, start_frame, end_frame, label_id)

        # Visualization
        st.divider()
        st.subheader("Current Annotations")
        
        # Save to Disk
        if st.button("💾 Save All Changes to Disk", type="primary"):
             msg = save_annotations_to_disk(st.session_state.annotations, LABELS_FILE, st.session_state.video_lengths)
             st.success(msg)
             
        if selected_video_name in st.session_state.annotations:
            df = pd.DataFrame(st.session_state.annotations[selected_video_name])
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No annotations for this video yet.")
            
        # Global view
        with st.expander("View All Session Data"):
             st.json(st.session_state.annotations)

elif page == "Training":
    st.header("Model Training")
    st.write("Train the FERAL model on annotated data.")
    
    # We would need a button to "Commit Annotations" which writes the JSON
    # For now, we assume the JSON exists.
    
    if st.button("Start Training"):
        pid = backend.run_training_background("../re_encoded_train", "../feral_behavioral_labels.json")
        st.success(f"Training started! PID: {pid}. Monitor 'dashboard_train.log' for progress.")

elif page == "Inference":
    st.header("Inference")
    st.write("Run the model on new raw videos.")
    video_files = glob.glob(os.path.join(RAW_VIDEOS_DIR, "*.mp4"))
    
    if not video_files:
        st.warning("No videos in raw_videos.")
    else:
        selected_video_name = st.selectbox("Select Video for Inference", [os.path.basename(v) for v in video_files])
        selected_video_path = os.path.join(RAW_VIDEOS_DIR, selected_video_name)
        
        if st.button("Run Inference"):
            with st.spinner(f"Running inference on {selected_video_name}... this may take a minute"):
                result = backend.run_inference(selected_video_path)
                if "Error" in result or "Exception" in result:
                    st.error(result)
                else:
                    st.success(result)

