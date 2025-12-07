import streamlit as st
import os
import glob
import json
import backend  # Import our new backend module

# Config
DATA_DIR = "../"
RAW_VIDEOS_DIR = os.path.join(DATA_DIR, "raw_videos")
TRAINING_DIR = os.path.join(DATA_DIR, "training_data")
LABELS_FILE = os.path.join(DATA_DIR, "feral_behavioral_labels.json")

if "annotations" not in st.session_state:
    st.session_state.annotations = {}

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
        
        st.video(selected_video_path)
        
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
        
        start_frame, end_frame = segment_range
        st.caption(f"Selected Segment: Frames {start_frame} to {end_frame}")

        label_id = st.number_input("Class ID", min_value=0, value=1)
            
        if st.button("Add Label Segment"):
            save_annotation(selected_video_name, start_frame, end_frame, label_id)
            
        st.write("Current Session Annotations:")
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

