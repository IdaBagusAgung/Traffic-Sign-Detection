import streamlit as st
import cv2
from PIL import Image
from ultralytics import YOLO
import numpy as np

st.title("Deteksi Objek dengan YOLO")

# Initialize session state for the video capture control
if 'run' not in st.session_state:
    st.session_state.run = False

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO('runs/detect/train12/weights/best.pt')

model = load_model()

# Function to perform object detection
def detect_objects(image):
    result = model.predict(image)
    return result[0].plot()

# Function to capture video from webcam
def capture_video():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise Exception("No Camera")
    
    stframe = st.empty()
    while st.session_state.run:
        ret, frame = cam.read()
        if not ret:
            break
        
        # Resize frame for faster processing
        frame_resized = cv2.resize(frame, (640, 480))
        
        # Perform object detection
        result_img = detect_objects(frame_resized)
        
        # Convert the result image from OpenCV format to RGB format for Streamlit
        result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        
        # Display the detected objects
        stframe.image(result_img_rgb, caption='Detected Objects', use_column_width=True)
        
        # Add a small delay to prevent overloading the CPU
        cv2.waitKey(1)
    
    # Release the camera
    cam.release()

# Run the application
if __name__ == '__main__':
    st.sidebar.header("Deteksi Objek dengan YOLO")
    st.sidebar.write("Aplikasi ini menggunakan model YOLO untuk mendeteksi objek dalam video dari kamera.")
    
    # Start button
    start_button = st.sidebar.button("Start")
    if start_button:
        st.session_state.run = True
        capture_video()
    
    # Stop button
    stop_button = st.sidebar.button("Stop")
    if stop_button:
        st.session_state.run = False
