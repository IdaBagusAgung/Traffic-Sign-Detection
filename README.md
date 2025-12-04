# Traffic Sign Detection

This repository contains the implementation of a traffic sign detection system using two different approaches: Convolutional Neural Network (CNN) and YOLO (You Only Look Once).

## 📁 Project Structure

### 1. CNN MODEL
This folder contains the implementation of traffic sign detection using classic CNN approach with various architectures:
- **Model Architecture**: MobileNet, VGG16, and Custom CNN
- **Dataset**: Indonesian traffic sign dataset (cropped and non-cropped)
- **Features**:
  - Training and testing with various configurations
  - Dashboard for result visualization
  - Models in `.h5` and `.keras` formats
  - Class indices for traffic sign category mapping

**Main Files:**
- `DSP-CNN.ipynb` - CNN model training notebook
- `Dashboard.py` - Dashboard for testing and visualization
- `model_rambu_lalu_lintas.h5` - Trained CNN model

### 2. YOLO MODEL
This folder contains the implementation of traffic sign detection using YOLOv8:
- **Model**: YOLOv8n (Nano)
- **Dataset**: YOLO format (images + labels in txt format)
- **Features**:
  - Real-time object detection
  - Multiple training experiments (train1-train1221)
  - Testing and validation
  - Integrated dashboard

**Main Files:**
- `DSP-YOLO.ipynb` - YOLOv8 training notebook
- `Dashboard.py` - Dashboard for testing
- `DashboardGabung.py` - Combined CNN and YOLO dashboard
- `data.yaml` - YOLO dataset configuration
- `runs/` - Training and detection results

## 🚀 How to Use

### CNN Model
1. Open `CNN MODEL/DSP-CNN.ipynb` for training
2. Run `Dashboard.py` for testing with the trained model
3. Models are saved in the `MODEL/` folder

### YOLO Model
1. Open `YOLO MODEL/DSP-YOLO.ipynb` for YOLOv8 training
2. Use `Dashboard.py` or `DashboardGabung.py` for testing
3. Training results are saved in the `runs/detect/` folder

## 📊 Dataset

The dataset contains various categories of Indonesian traffic signs:
- Traffic lights (green, yellow, red)
- Prohibition signs (no right/left turn, no parking, no stopping, etc.)
- Warning signs (zebra crossing, traffic signals, etc.)
- Direction and command signs

## 🛠️ Requirements

```
tensorflow
keras
ultralytics
opencv-python
numpy
pandas
matplotlib
streamlit (for dashboard)
```

## 📝 Notes

- Model files (`.h5`, `.keras`, `.pt`) and large datasets may not be uploaded to the repository
- To use the models, download or retrain using the available notebooks
- Dashboard can be run using Streamlit

## 👥 Author

Ida Bagus Agung

## 📄 License

Educational Project - Data Science Programming
