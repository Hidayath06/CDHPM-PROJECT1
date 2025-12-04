#app
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import plotly.graph_objects as go
import plotly.express as px
import warnings
import requests
import tempfile
#import cv2 # Included but not used in the core logic, keeping for completeness
import shutil # Included but not used in the core logic, keeping for completeness

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
# ⚠️ 1. REPLACE THIS LINK with your actual public download URL for the .h5 file!
MODEL_URL = "https://huggingface.co/Hidayathulla06/eczema-detector-resnet50/resolve/main/best_transfer_model_compatible.h5" 
LOCAL_MODEL_PATH = "downloaded_model_compatible.h5" 
TARGET_CLASSES = ['Eczema', 'Non-Eczema'] 
# -----------------------------------

# Set page config and custom CSS
st.set_page_config(page_title="Eczema Detection System", page_icon="🏥", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for styling (Example structure, replace '...' with your actual CSS)
st.markdown("""
<style>
    .main-header {
        font-size: 2.5em;
        color: #007BFF;
        text-align: center;
        margin-bottom: 20px;
    }
    .st-emotion-cache-1c9vsl {
        padding-top: 2rem;
    }
    /* Add more of your custom CSS here */
</style>
""", unsafe_allow_html=True) 

# --- MODEL LOADING (FIXED) ---

@st.cache_resource
def load_model_from_url():
    """
    Downloads the Keras model from the URL and loads it.
    Returns the loaded Keras model object.
    """
    # Use a temporary directory path for safe local storage
    local_path = os.path.join(tempfile.gettempdir(), LOCAL_MODEL_PATH)
    
    # 1. Download the Model
    if not os.path.exists(local_path):
        try:
            st.info(f"Downloading model from Hugging Face...")
            # Use a longer timeout for large model files
            response = requests.get(MODEL_URL, stream=True, timeout=600) 
            response.raise_for_status() 

            # Write content directly to the temporary file
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            st.success("Model downloaded successfully!")

        except Exception as e:
            st.error("Model download failed. Check the URL and try again.")
            st.error(f"Details: Download/Save Error: {e}")
            return None
    
    # 2. Load the Model from Local Disk
    try:
        # Load the Keras model directly
        model = tf.keras.models.load_model(local_path)
        st.success("Model initialized and ready!")
        return model
        
    except Exception as e:
        st.error("Model loading failed. The downloaded file may be corrupted.")
        st.error(f"Details: Load Error: {e}")
        # Optionally, delete the potentially corrupted file to force a re-download next run
        if os.path.exists(local_path):
            os.remove(local_path)
            st.warning("Potentially corrupted file removed from cache. Try clearing Streamlit cache too.")
        return None

# --- HELPER FUNCTIONS ---

def preprocess_image(image):
    """Resizes and normalizes the PIL image for model prediction."""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Model's expected input size
    image = image.resize((224, 224))
    
    # Convert to array and normalize
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def get_confidence_color(confidence):
    """Returns a color based on confidence level for visualization."""
    if confidence >= 0.85:
        return "green"
    elif confidence >= 0.65:
        return "orange"
    else:
        return "red"

def create_gauge_chart(confidence, class_name):
    """Creates a Plotly gauge chart for confidence visualization."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = confidence * 100,
        title = {'text': f"Confidence: {class_name}", 'font': {'size': 24}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': get_confidence_color(confidence)},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 65], 'color': 'rgba(255, 99, 71, 0.5)'},   # Red/Low
                {'range': [65, 85], 'color': 'rgba(255, 165, 0, 0.5)'},  # Orange/Moderate
                {'range': [85, 100], 'color': 'rgba(60, 179, 113, 0.5)'}], # Green/High
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90}}))
    
    fig.update_layout(height=300, margin=dict(t=10, b=10, l=10, r=10))
    return fig

# --- MAIN APPLICATION LOGIC ---

def main():
    st.markdown('<h1 class="main-header">🏥 Eczema Detection System</h1>', unsafe_allow_html=True)
    
    st.sidebar.markdown('## Navigation')
    page = st.sidebar.selectbox("Choose a page", ["Home", "Upload & Predict", "Model Information"])

    # Load the Keras model (cached)
    model_instance = load_model_from_url() 
    
    if page == "Home":
        st.markdown("""
            ### Welcome to the Eczema Detection System 👋
            
            This application uses a deep learning model (ResNet50 transfer learning) to classify skin images into two categories: **Eczema** or **Non-Eczema**.
            
            * **Purpose:** To provide a quick preliminary assessment based on an image.
            * **Disclaimer:** This system is for informational and educational purposes only and is **not a substitute for professional medical advice, diagnosis, or treatment.** Always consult a qualified healthcare provider for any health concerns.
        """)
        
    elif page == "Upload & Predict":
        st.header("📤 Upload Image for Eczema Diagnosis")
        
        if model_instance is None:
            st.error("🚨 Application cannot run because the model failed to load. Please check the logs above.")
            st.stop()
        
        uploaded_file = st.file_uploader(
            "Choose an image file", type=['png', 'jpg', 'jpeg'], 
            help="Upload a clear image of the affected skin area (PNG, JPG, or JPEG format)."
        )
        
        if uploaded_file is not None:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📷 Uploaded Image")
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
            
            with col2:
                st.subheader("🔍 Analysis Results")
                
                # Preprocess the image
                img_array = preprocess_image(image)
                
                with st.spinner("Analyzing image..."):
                    # Use the Keras model instance directly
                    prediction = model_instance.predict(img_array)
                    
                    # Assuming the model outputs probabilities for TARGET_CLASSES
                    probabilities = prediction[0][:len(TARGET_CLASSES)] 
                    predicted_class_index = np.argmax(probabilities)
                    
                    # Ensure confidence is a float (it is, but good practice)
                    confidence = float(probabilities[predicted_class_index]) 
                    disease_name = TARGET_CLASSES[predicted_class_index]
                
                # --- Display Results ---
                
                # Display the main result
                confidence_percent = f"{confidence * 100:.2f}%"
                st.markdown(f"""
                    ### Predicted Class: <span style='color:{get_confidence_color(confidence)};'>**{disease_name}**</span>
                    #### Confidence: **{confidence_percent}**
                """, unsafe_allow_html=True)

                # Display gauge chart
                fig_gauge = create_gauge_chart(confidence, disease_name)
                st.plotly_chart(fig_gauge, use_container_width=True)

                # Show all probabilities in a bar chart
                df_results = px.data.gapminder().head(len(TARGET_CLASSES)).copy()
                df_results['Disease'] = TARGET_CLASSES
                df_results['Probability'] = [float(p * 100) for p in probabilities]
                
                fig_bar = px.bar(
                    df_results, 
                    x='Disease', 
                    y='Probability', 
                    color='Disease',
                    title='Class Probabilities',
                    height=300
                )
                fig_bar.update_layout(showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True)
                
                st.warning("⚠️ **Medical Disclaimer:** This is an AI-generated result. Consult a dermatologist for an accurate diagnosis.")

elif page == "Model Information":
        st.header("⚙️ Model Details & Project Information")
        st.markdown(f"""
            ## 🧠 Model Architecture & Performance
            ---
            The model uses **Transfer Learning** with the powerful ResNet50 architecture. This technique utilizes features pre-trained on the vast ImageNet dataset to efficiently classify skin conditions.

            * **Base Model:** ResNet50 (Residual Network with 50 layers)
            * **Transfer Learning:** Only the final classification head was trained; the 50 base layers were frozen.
            * **Input Size:** 224x224 pixels (RGB)
            * **Output Classes:** {', '.join(TARGET_CLASSES)}
            
            ### Final Test Set Metrics
            The model was rigorously evaluated on a completely unseen test set:
            
            * **Accuracy:** **98.40%** * **F1-Score (Eczema):** **0.98**
            * **F1-Score (Non-Eczema):** **0.98**
            
            ---
            
            ## 👨‍💻 Project Developer Information
            
            * **Developer Name:** **S A Hidayathulla**
            * **Project/System Name:** **CDHPM-PROJECT1: Eczema Detection System**
            * **College/Institution:** **The Apollo University**
            * **Academic Year:** **[2024-2025]**
            * **Technology Stack:** Python, TensorFlow/Keras, ResNet50, Streamlit, Plotly, Hugging Face
            * **GitHub Repository:** [https://github.com/Hidayath06/CDHPM-PROJECT1](https://github.com/Hidayath06/CDHPM-PROJECT1)
            * **Model File Location:** [Hugging Face Repository]({MODEL_URL})
            
            ### Technical Notes
            The use of data augmentation (rotation, flipping, zooming) was critical during training to improve the model's ability to generalize to real-world variations in skin images.
        """, unsafe_allow_html=True)
if __name__ == "__main__":
    main()




