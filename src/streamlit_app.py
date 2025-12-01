"""
Streamlit Web UI for Shoe Classification System
Provides visualization, prediction, and retraining interface
"""

import streamlit as st
import requests
import os
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import time
from io import BytesIO

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Page config
st.set_page_config(
    page_title="Shoe Classifier Dashboard",
    page_icon="👟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# Helper functions
def check_api_status():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def get_api_status():
    """Get detailed API status"""
    try:
        response = requests.get(f"{API_URL}/status", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def get_metrics():
    """Get model metrics"""
    try:
        response = requests.get(f"{API_URL}/metrics", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def predict_image(image_file):
    """Send image to API for prediction"""
    try:
        # Properly format the file for multipart upload
        files = {"file": (image_file.name, image_file.getvalue(), image_file.type)}
        response = requests.post(f"{API_URL}/predict", files=files, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API error: {response.status_code} - {response.text}"}
    except Exception as e:
        return {"error": str(e)}


def upload_training_data(files, category):
    """Upload training data to API"""
    try:
        file_list = [("files", (f.name, f, "image/jpeg")) for f in files]
        response = requests.post(
            f"{API_URL}/upload-data",
            files=file_list,
            params={"category": category},
            timeout=60
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Upload failed: {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def trigger_retrain():
    """Trigger model retraining"""
    try:
        response = requests.post(f"{API_URL}/retrain", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Retrain failed: {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def get_training_progress():
    """Get training progress"""
    try:
        response = requests.get(f"{API_URL}/training-progress", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


# Main app
def main():
    st.markdown('<h1 class="main-header">👟 Shoe Classification Dashboard</h1>', unsafe_allow_html=True)
    
    # Check API status
    if not check_api_status():
        st.error(" API is not running. Please start the API server first.")
        st.code("python src/app.py", language="bash")
        return
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        [" Home", " Predict", " Analytics", " Upload Data", " Retrain Model", " System Status"]
    )
    
    # Pages
    if page == " Home":
        show_home()
    elif page == " Predict":
        show_predict()
    elif page == " Analytics":
        show_analytics()
    elif page == " Upload Data":
        show_upload()
    elif page == " Retrain Model":
        show_retrain()
    elif page == " System Status":
        show_status()


def show_home():
    """Home page"""
    st.header("Welcome to Shoe Classification System")
    
    st.markdown("""
    This application uses deep learning to classify shoe images into three categories:
    -  **Boot**
    -  **Sandal**
    -  **Shoe**
    
    ### Features
    -  **Real-time Prediction**: Upload an image and get instant classification
    -  **Analytics Dashboard**: View model performance metrics and visualizations
    -  **Data Upload**: Upload new training data for model improvement
    -  **Model Retraining**: Retrain the model with new data
    -  **System Monitoring**: Track API uptime and performance
    
    
    """)
    
    # Quick stats
    status = get_api_status()
    if status:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("API Status", "🟢 Online")
        with col2:
            st.metric("Total Requests", status.get('total_requests', 0))
        with col3:
            st.metric("Uptime", status.get('uptime', 'N/A'))
        with col4:
            model_status = " yes Loaded" if status.get('model_loaded') else " Not Loaded"
            st.metric("Model Status", model_status)


def show_predict():
    """Prediction page"""
    st.header("🔮 Predict Shoe Type")
    
    st.markdown("Upload an image of a shoe to classify it.")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of a shoe"
    )
    
    if uploaded_file is not None:
        # Display image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader("Prediction Results")
            
            if st.button(" Classify", type="primary", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    # Reset file pointer
                    uploaded_file.seek(0)
                    result = predict_image(uploaded_file)
                    
                    if "error" in result:
                        st.error(f" {result['error']}")
                    else:
                        # Display prediction
                        st.success(f"### Predicted Class: **{result['predicted_class']}**")
                        st.metric("Confidence", f"{result['confidence']:.2%}")
                        
                        # Display probabilities
                        st.subheader("Class Probabilities")
                        probs = result['probabilities']
                        
                        # Create bar chart
                        fig = go.Figure(data=[
                            go.Bar(
                                x=list(probs.keys()),
                                y=list(probs.values()),
                                marker_color=['#FF6B6B', '#4ECDC4', '#95E1D3']
                            )
                        ])
                        fig.update_layout(
                            title="Prediction Probabilities",
                            xaxis_title="Class",
                            yaxis_title="Probability",
                            yaxis=dict(tickformat=".0%")
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show detailed probabilities
                        prob_df = pd.DataFrame({
                            'Class': list(probs.keys()),
                            'Probability': [f"{v:.2%}" for v in probs.values()]
                        })
                        st.table(prob_df)


def show_analytics():
    """Analytics page"""
    st.header(" Model Analytics")
    
    metrics = get_metrics()
    
    if metrics:
        st.success(" Model metrics loaded successfully!")
        
        # Display metrics in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}")
            st.metric("Loss", f"{metrics.get('loss', 0):.4f}")
        
        with col2:
            st.metric("Precision", f"{metrics.get('precision', 0):.2%}")
            st.metric("Recall", f"{metrics.get('recall', 0):.2%}")
        
        with col3:
            st.metric("F1-Score", f"{metrics.get('f1_score', 0):.2%}")
            st.metric("AUC", f"{metrics.get('auc', 0):.4f}")
        
        # Visualize metrics
        st.subheader("Performance Metrics Visualization")
        
        # Create metrics dataframe
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        metric_values = [
            metrics.get('accuracy', 0),
            metrics.get('precision', 0),
            metrics.get('recall', 0),
            metrics.get('f1_score', 0)
        ]
        
        # Bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=metric_names,
                y=metric_values,
                marker_color=['#FF6B6B', '#4ECDC4', '#95E1D3', '#FFA07A']
            )
        ])
        fig.update_layout(
            title="Model Performance Metrics",
            xaxis_title="Metric",
            yaxis_title="Value",
            yaxis=dict(tickformat=".0%", range=[0, 1])
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Radar chart
        fig = go.Figure(data=go.Scatterpolar(
            r=metric_values,
            theta=metric_names,
            fill='toself',
            line_color='#1f77b4'
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Performance Radar Chart"
        )
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.warning(" No metrics available. Train the model first.")


def show_upload():
    """Upload data page"""
    st.header(" Upload Training Data")
    
    st.markdown("""
    Upload images to expand the training dataset. You can upload files per category or let the system 
    auto-detect based on filename patterns (e.g., files with 'boot' in name go to Boot category).
    """)
    
    # Upload mode selection
    upload_mode = st.radio(
        "Upload Mode",
        ["Single Category", "Auto-detect from Filename"],
        help="Choose how to organize uploaded images"
    )
    
    if upload_mode == "Single Category":
        # Category selection
        category = st.selectbox(
            "Select Category",
            ["Boot", "Sandal", "Shoe"],
            help="All uploaded images will go to this category"
        )
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload images",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            help="Select multiple images to upload to the selected category"
        )
        
        if uploaded_files:
            st.info(f"📁 {len(uploaded_files)} files selected → **{category}** category")
            
            # Show preview
            if len(uploaded_files) <= 6:
                cols = st.columns(min(len(uploaded_files), 3))
                for idx, file in enumerate(uploaded_files):
                    with cols[idx % 3]:
                        image = Image.open(file)
                        st.image(image, caption=file.name, use_container_width=True)
            
            if st.button("📤 Upload to Server", type="primary", use_container_width=True):
                with st.spinner("Uploading files..."):
                    # Reset file pointers
                    for file in uploaded_files:
                        file.seek(0)
                    
                    result = upload_training_data(uploaded_files, category)
                    
                    if "error" in result:
                        st.error(f"❌ {result['error']}")
                    else:
                        st.success(f"✅ {result['message']}")
                        
                        # Show uploaded files in a nicer format
                        if 'saved_files' in result and result['saved_files']:
                            with st.expander("📁 View uploaded files", expanded=False):
                                for idx, filename in enumerate(result['saved_files'], 1):
                                    st.text(f"{idx}. {filename}")
    
    else:  # Auto-detect mode
        st.info("💡 Files will be categorized based on their names (e.g., 'boot_01.jpg' → Boot)")
        
        uploaded_files = st.file_uploader(
            "Upload images (auto-detect category)",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            help="Filenames should contain 'boot', 'sandal', or 'shoe'"
        )
        
        if uploaded_files:
            # Auto-detect categories from filenames
            categorized = {"Boot": [], "Sandal": [], "Shoe": [], "Unknown": []}
            
            for file in uploaded_files:
                filename_lower = file.name.lower()
                if 'boot' in filename_lower:
                    categorized["Boot"].append(file)
                elif 'sandal' in filename_lower:
                    categorized["Sandal"].append(file)
                elif 'shoe' in filename_lower:
                    categorized["Shoe"].append(file)
                else:
                    categorized["Unknown"].append(file)
            
            # Show categorization summary
            st.markdown("#### 📊 Detected Categories:")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Boot", len(categorized["Boot"]))
            with col2:
                st.metric("Sandal", len(categorized["Sandal"]))
            with col3:
                st.metric("Shoe", len(categorized["Shoe"]))
            with col4:
                st.metric("Unknown", len(categorized["Unknown"]))
            
            if categorized["Unknown"]:
                st.warning(f"⚠️ {len(categorized['Unknown'])} files couldn't be auto-categorized. They will be skipped.")
                with st.expander("View unknown files"):
                    for file in categorized["Unknown"]:
                        st.text(f"• {file.name}")
            
            if st.button("📤 Upload to Server", type="primary", use_container_width=True):
                with st.spinner("Uploading files..."):
                    upload_results = []
                    
                    for category, files in categorized.items():
                        if category != "Unknown" and files:
                            # Reset file pointers
                            for file in files:
                                file.seek(0)
                            
                            result = upload_training_data(files, category)
                            upload_results.append((category, result))
                    
                    # Display results
                    for category, result in upload_results:
                        if "error" in result:
                            st.error(f"❌ {category}: {result['error']}")
                        else:
                            st.success(f"✅ {category}: {result['message']}")


def show_retrain():
    """Retrain model page"""
    st.header(" Retrain Model")
    
    st.markdown("""
    Retrain the model using uploaded data to improve its performance.
    
     **Note**: Retraining may take several minutes depending on the amount of data.
    """)
    
    # Check training status
    progress = get_training_progress()
    
    if progress and progress.get('is_training'):
        st.warning(" Training in progress...")
        
        # Show progress bar
        progress_value = progress.get('progress', 0)
        st.progress(progress_value / 100)
        st.info(f" Progress: {progress_value}% - {progress.get('message', 'Processing...')}")
        
        # Auto-refresh
        time.sleep(2)
        st.rerun()
        
    else:
        if st.button(" Start Retraining", type="primary", use_container_width=True):
            result = trigger_retrain()
            
            if "error" in result:
                st.error(f"❌ {result['error']}")
            else:
                st.success(f"✅ {result['message']}")
                time.sleep(1)
                st.rerun()
        
        # Show last training info
        if progress and progress.get('last_trained'):
            st.info(f" Last trained: {progress['last_trained']}")


def show_status():
    """System status page"""
    st.header(" System Status")
    
    status = get_api_status()
    
    if status:
        # Main status
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("API Status")
            st.success("🟢 Online")
            st.metric("Uptime", status.get('uptime', 'N/A'))
            st.metric("Total Requests", status.get('total_requests', 0))
        
        with col2:
            st.subheader("Model Status")
            model_loaded = status.get('model_loaded', False)
            if model_loaded:
                st.success(" Model Loaded")
            else:
                st.error(" Model Not Loaded")
        
        # Training status
        st.subheader("Training Status")
        train_status = status.get('training_status', {})
        
        if train_status.get('is_training'):
            st.warning(f" Training in Progress: {train_status.get('progress', 0)}%")
        else:
            st.info(f" {train_status.get('message', 'Idle')}")
        
        if train_status.get('last_trained'):
            st.info(f" Last Training: {train_status['last_trained']}")
        
        # Model metrics
        if status.get('model_metrics'):
            st.subheader("Current Model Metrics")
            metrics_df = pd.DataFrame([status['model_metrics']])
            st.dataframe(metrics_df, use_container_width=True)
        
    else:
        st.error(" Unable to connect to API")


if __name__ == "__main__":
    main()
