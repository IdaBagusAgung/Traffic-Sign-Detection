import streamlit as st
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import json
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras import metrics

# [server]
# enableXsrfProtection=false

# Load the model
model_path = 'model_rambu_lalu_lintas.h5'
if not os.path.exists(model_path):
    st.error(f"Model file '{model_path}' not found. Please make sure the model is available.")
    st.stop()

model = load_model(model_path)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Perform a dummy evaluation to initialize metrics
dummy_x = np.zeros((1, 224, 224, 3))
dummy_y = np.zeros((1,))
model.evaluate(dummy_x, dummy_y, verbose=0)

# Load class indices from JSON file
class_indices_path = 'class_indices.json'
if not os.path.exists(class_indices_path):
    st.error(f"Class indices file '{class_indices_path}' not found. Please make sure the file is available.")
    st.stop()

with open(class_indices_path, 'r') as f:
    class_indices = json.load(f)

# Function to detect and crop traffic sign
def detect_and_crop_traffic_sign_v2(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    lower_yellow = np.array([18, 70, 50])
    upper_yellow = np.array([35, 255, 255])

    lower_blue = np.array([90, 70, 50])
    upper_blue = np.array([128, 255, 255])

    lower_green = np.array([35, 70, 50])
    upper_green = np.array([85, 255, 255])

    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = mask_red1 + mask_red2

    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    mask = mask_red + mask_yellow + mask_blue + mask_green

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        contours = [c for c in contours if cv2.contourArea(c) > 500]

        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)

            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(image.shape[1], x + w), min(image.shape[0], y + h)
            cropped_image = image[y1:y2, x1:x2]
            return cv2.resize(cropped_image, (224, 224)), image
    return cv2.resize(image, (224, 224)), image

def preprocess_image_v2(img):
    img_array = img_to_array(img)
    return img_array

def predict_image_v2(img_array):
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)
    return predicted_class, prediction

def main():
    st.title("Traffic Sign Detection and Classification")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            st.write("File uploaded successfully!")
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            st.write("Image opened successfully!")

            if image.mode != 'RGB':
                image = image.convert('RGB')

            img_cv = np.array(image)

            cropped_img, image_with_bbox = detect_and_crop_traffic_sign_v2(img_cv)
            st.write("Traffic sign detected and cropped!")

            preprocessed_img = preprocess_image_v2(cropped_img)
            st.write("Image preprocessed!")

            predicted_class, prediction = predict_image_v2(preprocessed_img)
            st.write("Prediction made!")

            predicted_label = class_indices.get(str(predicted_class[0]), "Unknown")

            st.write(f"Predicted Label: {predicted_label}")
            st.write(f"Prediction Confidence: {prediction[0][predicted_class[0]] * 100:.2f}%")

            st.subheader("Preprocessed Image and Detected Traffic Sign")
            col1, col2 = st.columns(2)

            with col1:
                st.image(cropped_img, caption='Preprocessed Image', use_column_width=True)

            with col2:
                st.image(image_with_bbox, caption='Original Image with Bounding Box', use_column_width=True)

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
