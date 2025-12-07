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
# Default classes
if "class_names" not in st.session_state:
    st.session_state.class_names = {1: "Sleep", 2: "Antenna"}

def load_annotations():
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, "r") as f:
            data = json.load(f)
            if "class_names" in data:
                # Convert string keys to int
                st.session_state.class_names = {int(k): v for k, v in data["class_names"].items() if k != "0"} 

# Initial Load
if "loaded" not in st.session_state:
    load_annotations()
    st.session_state.loaded = True

def save_annotation(video_name, start_frame, end_frame, label_id):
    if video_name not in st.session_state.annotations:
        st.session_state.annotations[video_name] = []
    
    st.session_state.annotations[video_name].append({
        "start": start_frame,
        "end": end_frame,
        "label": label_id
    })
    cls_name = st.session_state.class_names.get(label_id, f"Class {label_id}")
    st.success(f"Added: {cls_name} ({start_frame}-{end_frame})")

st.set_page_config(page_title="FERAL Dashboard", layout="wide")
st.title("FERAL: Behavioral Analysis Dashboard")

with st.sidebar:
    st.header("Navigation")
    page = st.radio("Go to", ["Annotation", "Training", "Inference"])
    
    st.divider()
    st.header("Class Manager")
    # Add new class
    new_cls_name = st.text_input("New Class Name")
    # Find next available ID
    existing_ids = st.session_state.class_names.keys()
    next_id = max(existing_ids) + 1 if existing_ids else 1
    new_cls_id = st.number_input("New Class ID", value=next_id, min_value=1)
    
    if st.button("Add Class"):
        if new_cls_name:
            st.session_state.class_names[new_cls_id] = new_cls_name
            st.success(f"Added {new_cls_name} ({new_cls_id})")
        else:
            st.error("Enter a name.")
            
    st.write("Current Classes:")
    st.json(st.session_state.class_names)

# ... inside Annotation Page ...
        
        # Quick Class Buttons
        st.subheader("Class Selection")
        
        # Dynamic Columns
        classes = st.session_state.class_names
        cols = st.columns(len(classes) + 1)
        
        for idx, (cls_id, cls_name) in enumerate(classes.items()):
            with cols[idx]:
                if st.button(f"{cls_name} ({cls_id})"):
                    save_annotation(selected_video_name, start_frame, end_frame, cls_id)
                    
        with cols[-1]:
             label_id = st.number_input("Custom ID", min_value=0, value=1, label_visibility="collapsed")
             if st.button("Add ID"):
                  save_annotation(selected_video_name, start_frame, end_frame, label_id)

        # Visualization
        st.divider()
        st.subheader("Current Annotations")
        
        # Save to Disk
        if st.button("💾 Save All Changes to Disk", type="primary"):
             msg = save_annotations_to_disk(
                 st.session_state.annotations, 
                 LABELS_FILE, 
                 st.session_state.video_lengths,
                 st.session_state.class_names
             )
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
                    
        # Visualization
        st.subheader("Inference Results")
        results = backend.load_latest_inference_result()
        
        if results and selected_video_name in results:
             preds = results[selected_video_name] # list of ints
             if preds:
                  # Create a dataframe for Altair
                  # frame, class
                  df_res = pd.DataFrame({
                      "frame": range(len(preds)),
                      "class": preds
                  })
                  
                  import altair as alt
                  
                  chart = alt.Chart(df_res).mark_rect().encode(
                      x=alt.X('frame:Q', bin=alt.Bin(maxbins=100)), # Bin for raster-like view if long
                      y=alt.Y('class:O'),
                      color=alt.Color('class:N', scale=alt.Scale(scheme='category10')),
                      tooltip=['frame', 'class']
                  ).properties(
                      title=f"Behavioral Segmentation for {selected_video_name}",
                      width=700,
                      height=150
                  )
                  
                  st.altair_chart(chart, use_container_width=True)
                  
                  st.success(f"Visualizing {len(preds)} frames of predictions.")
             else:
                  st.warning("Empty predictions found.")
        elif results:
             st.info(f"No results found for {selected_video_name} (found results for: {list(results.keys())})")
        else:
             st.info("No inference results found. Run inference first.")

