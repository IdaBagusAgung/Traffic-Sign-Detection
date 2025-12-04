# DSP-PROJECT - Traffic Sign Detection with YOLOv8

[![Made by](https://img.shields.io/badge/Made%20by-Gus%20Agung%20Dev-blue)](https://github.com/gusagung)
[![Last Updated](https://img.shields.io/badge/Last%20Updated-December%204%2C%202025-green)]()
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-red)](https://ultralytics.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)](https://pytorch.org/)

## 📋 Description

This project implements an Indonesian traffic sign detection system using YOLOv8 (You Only Look Once version 8). Unlike classification in DSP CNN, this project focuses on **object detection** which can detect multiple signs in one image while providing the location (bounding box) of each sign.

## 🎯 Project Objectives

1. Detect and localize traffic signs in images
2. Identify multiple objects in one frame
3. Real-time detection with high accuracy
4. Achieve high precision and recall
5. Build an interactive dashboard for testing

## 🌟 Key Features

- ✅ **Real-time Detection**: Deteksi cepat dengan YOLOv8
- ✅ **Multiple Object Detection**: Deteksi banyak rambu sekaligus
- ✅ **Bounding Box Visualization**: Box dengan confidence score
- ✅ **High Accuracy**: mAP50 > 87%
- ✅ **Interactive Dashboard**: Upload dan prediksi via web
- ✅ **Comprehensive Metrics**: Precision, Recall, mAP
- ✅ **Flexible Input**: Support image dan video

## 📁 Struktur Folder

```
DSP-PROJECT/
├── Dashboard.py                 # Streamlit dashboard YOLO only
├── DashboardGabung.py          # Dashboard CNN + YOLO
├── DSP-YOLO.ipynb              # Main training notebook
├── DSP-YOLO-OLD.ipynb          # Previous experiments
├── YOLO_TESTING_1.ipynb        # Testing notebook
├── testing.py                   # Python testing script
├── data.yaml                    # Dataset configuration
├── yolov8n.pt                  # Pretrained YOLOv8 nano
├── model_rambu_lalu_lintas.h5  # CNN model (untuk dashboard gabung)
├── class_indices.json          # Class mapping
│
├── train/                      # Training dataset
│   ├── images/                 # Training images
│   │   ├── 1-1-_jpg.rf.xxx.jpg
│   │   └── ...
│   └── labels/                 # YOLO format labels (.txt)
│       ├── 1-1-_jpg.rf.xxx.txt
│       └── ...
│
├── test/                       # Testing dataset
│   ├── images/                 # Test images
│   └── labels/                 # Test labels
│
├── valid/                      # Validation dataset
│   ├── images/                 # Validation images
│   └── labels/                 # Validation labels
│
├── testing-input-image/        # Custom test images
├── testing-yolo/              # YOLO test outputs
│
└── runs/                       # Training outputs
    └── detect/
        ├── train/              # Initial training
        ├── train2/ to train12/ # Training iterations
        │   └── train12/        # Best model (100 epochs)
        │       ├── weights/
        │       │   ├── best.pt       # Best checkpoint
        │       │   └── last.pt       # Last checkpoint
        │       ├── results.csv       # Training metrics
        │       ├── confusion_matrix.png
        │       ├── P_curve.png       # Precision curve
        │       ├── R_curve.png       # Recall curve
        │       └── ...
        ├── val/                # Validation results
        └── hasil-testing/      # Prediction outputs
            └── testing-input-image/
```

## 🛠️ Requirements

### System Requirements
- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM
- 10GB+ free disk space

### Dependencies
```txt
ultralytics>=8.0.0
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
opencv-python>=4.5.0
matplotlib>=3.4.0
seaborn>=0.11.0
pandas>=1.3.0
pillow>=8.3.0
streamlit>=1.10.0
ipywidgets>=7.6.0
tqdm>=4.62.0
```

### Installation
```bash
# Install ultralytics (includes all dependencies)
pip install ultralytics

# Or install individually
pip install torch torchvision opencv-python matplotlib pandas pillow streamlit ipywidgets
```

## 📊 YOLOv8 Architecture

### Model Information
```
Model: YOLOv8n (Nano)
- Base: yolov8n.pt (pretrained on COCO)
- Input Size: 416x416
- Parameters: ~3.2M
- Speed: ~80 FPS (on GPU)
- Size: ~6MB
```

### Custom Configuration
```python
model = YOLO('yolov8n.yaml').load('yolov8n.pt')

# Training parameters
results = model.train(
    data='data.yaml',
    epochs=100,
    imgsz=416,
    batch=64,
    lr0=0.0001,      # Initial learning rate
    dropout=0.15,    # Dropout rate
    amp=False,       # Automatic Mixed Precision
    device=0         # GPU device
)
```

## 📋 Dataset Configuration

### data.yaml Structure
```yaml
train: train/images
val: valid/images
test: test/images

nc: 15  # Number of classes
names: ['lampu-hijau', 'lampu-kuning', 'lampu-merah', 
        'larangan-parkir', 'larangan-berhenti', ...]
```

### YOLO Label Format
```txt
# Format: class_id x_center y_center width height (normalized 0-1)
0 0.5 0.5 0.3 0.4
1 0.7 0.3 0.2 0.25
```

### Label Generation
Labels must be in YOLO format:
- class_id: category index (0-indexed)
- x_center, y_center: center coordinates (normalized)
- width, height: box size (normalized)

## 🚀 Training Process

### 1. Dataset Preparation
```python
# Dataset structure is already correct
# Make sure data.yaml points to the correct path
```

### 2. Training Command
```python
from ultralytics import YOLO

# Load pretrained model
model = YOLO('yolov8n.pt')

# Train
results = model.train(
    data='data.yaml',
    epochs=100,
    imgsz=416,
    batch=64,
    device=0
)
```

### 3. Training Notebooks

#### DSP-YOLO.ipynb - Main Training
Main notebook contains:
- ✅ Dataset setup and Google Drive mounting
- ✅ Import libraries
- ✅ Dataset exploration and visualization
- ✅ Pretrained model testing
- ✅ GPU checking
- ✅ Full training pipeline (100 epochs)
- ✅ Confusion matrix analysis
- ✅ Training curves visualization
- ✅ Model evaluation
- ✅ Testing with test dataset
- ✅ Interactive image upload widget

#### YOLO_TESTING_1.ipynb - Testing
Notebook for testing:
- Testing on custom images
- Batch prediction
- Confidence threshold tuning
- IoU threshold tuning

### 4. Training Iterations
This project performs iterative training:
- **train1-train11**: Initial experiments (various hyperparameters)
- **train12**: Best model (100 epochs, optimal hyperparameters)
- **train1210-train1221**: Fine-tuning experiments

### 5. Training Results (train12)
```
Training: 100 epochs
Image Size: 416x416
Batch Size: 64
Learning Rate: 0.0001
Dropout: 0.15
```

## 📈 Performance Metrics

### Model Evaluation Results

| Metric | Value | Description |
|--------|-------|-------------|
| **Precision** | 0.85-0.90 | Akurasi prediksi positif |
| **Recall** | 0.82-0.88 | Coverage deteksi objek |
| **mAP50** | 0.87-0.92 | Mean Average Precision @ IoU=0.5 |
| **mAP50-95** | 0.65-0.75 | mAP @ IoU=0.5:0.95 |

### Training Metrics
- **Box Loss**: Converged ~0.8
- **Class Loss**: Converged ~0.5
- **DFL Loss**: Converged ~0.9

### Per-Class Performance
Best performing classes:
- Red Light: ~95% precision
- No Parking: ~92% precision
- No Stopping: ~90% precision

## 📊 Visualization & Analysis

### Available Plots
1. **Confusion Matrix** (`confusion_matrix.png`)
   - Class-wise prediction analysis
   - True positives, false positives visualization

2. **Precision Curve** (`P_curve.png`)
   - Precision across confidence thresholds
   - Per-class precision analysis

3. **Recall Curve** (`R_curve.png`)
   - Recall across confidence thresholds
   - Detection coverage per class

4. **Training Curves** (from `results.csv`)
   - Train/Val Box Loss
   - Train/Val Class Loss
   - Train/Val DFL Loss
   - Precision, Recall, mAP curves over epochs

### Generating Visualizations
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load results
results = pd.read_csv('runs/detect/train12/results.csv')

# Plot training metrics
plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
plt.plot(results['epoch'], results['train/box_loss'])
plt.title('Training Box Loss')
# ... more plots
```

## 🖥️ Dashboard Applications

### 1. YOLO-Only Dashboard
```bash
streamlit run Dashboard.py
```

**Features:**
- Upload single image
- Real-time YOLO detection
- Bounding boxes with labels
- Confidence scores
- Save results

### 2. Combined Dashboard (CNN + YOLO)
```bash
streamlit run DashboardGabung.py
```

**Features:**
- Select model: CNN Classification or YOLO Detection
- CNN: Single object classification
- YOLO: Multiple object detection
- Side-by-side comparison
- Detailed predictions

### Dashboard Usage
```python
# Upload image via browser
# Select model (CNN/YOLO)
# View predictions with:
#   - Bounding boxes
#   - Class labels
#   - Confidence scores
# Download results
```

## 🧪 Testing & Prediction

### 1. Testing on Test Dataset
```python
# Load best model
model = YOLO('runs/detect/train12/weights/best.pt')

# Run validation
metrics = model.val(split='test')

# Print metrics
print(f"Precision: {metrics.results_dict['metrics/precision(B)']}")
print(f"Recall: {metrics.results_dict['metrics/recall(B)']}")
print(f"mAP50: {metrics.results_dict['metrics/mAP50(B)']}")
```

### 2. Batch Prediction
```python
import os
import random
import cv2

image_dir = 'test/images'
output_dir = 'runs/detect/hasil-testing'

# Load model
model = YOLO('runs/detect/train12/weights/best.pt')

# Process images
for image_path in image_files:
    image = cv2.imread(image_path)
    results = model.predict(
        source=image, 
        imgsz=416, 
        conf=0.5,  # Confidence threshold
        iou=0.7     # IoU threshold
    )
    
    # Save results
    for result in results:
        result_image = result.plot()
        cv2.imwrite(output_path, result_image)
```

### 3. Interactive Testing (Jupyter)
```python
import ipywidgets as widgets
from IPython.display import display

# Create upload widget
upload_widget = widgets.FileUpload(
    accept='image/*', 
    multiple=False
)

def predict_on_upload(change):
    uploaded_file = change['new'][0]
    # ... prediction logic
    
upload_widget.observe(predict_on_upload, names='value')
display(upload_widget)
```

### 4. Custom Image Testing
```python
# Single image prediction
model = YOLO('runs/detect/train12/weights/best.pt')
results = model.predict('path/to/image.jpg')

# Display results
results[0].show()

# Access detection data
boxes = results[0].boxes.xyxy  # Bounding boxes
scores = results[0].boxes.conf  # Confidence scores
classes = results[0].boxes.cls  # Class IDs
```

## 🔧 Advanced Configuration

### Inference Parameters
```python
results = model.predict(
    source='image.jpg',
    imgsz=416,          # Input size
    conf=0.5,           # Confidence threshold
    iou=0.7,            # IoU threshold for NMS
    max_det=100,        # Max detections per image
    device=0,           # GPU device
    save=True,          # Save results
    save_txt=True,      # Save labels
    save_conf=True      # Save confidence
)
```

### Post-Processing
```python
# Filter by confidence
high_conf_results = results[results.boxes.conf > 0.8]

# Filter by class
specific_class = results[results.boxes.cls == 0]  # Class 0 only

# Non-Maximum Suppression (already applied by YOLO)
# But can be adjusted via 'iou' parameter
```

## 📱 Model Export

### Export to Different Formats
```python
# Export to ONNX
model.export(format='onnx')

# Export to TensorFlow Lite
model.export(format='tflite')

# Export to CoreML (for iOS)
model.export(format='coreml')

# Export to TensorFlow SavedModel
model.export(format='saved_model')
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```python
   # Reduce batch size
   batch=32  # instead of 64
   
   # Reduce image size
   imgsz=320  # instead of 416
   ```

2. **Low mAP**
   - Check label quality
   - Increase training epochs
   - Try different image size
   - Adjust learning rate
   - Add more training data

3. **False Positives**
   ```python
   # Increase confidence threshold
   results = model.predict(source='img.jpg', conf=0.7)
   ```

4. **Missed Detections**
   ```python
   # Lower confidence threshold
   results = model.predict(source='img.jpg', conf=0.3)
   
   # Adjust IoU threshold
   results = model.predict(source='img.jpg', iou=0.5)
   ```

## 📝 Best Practices

### Training
- ✅ Use GPU for training
- ✅ Start with pretrained weights
- ✅ Monitor training metrics regularly
- ✅ Save checkpoints frequently
- ✅ Validate on separate dataset
- ✅ Use data augmentation (built-in in YOLO)

### Inference
- ✅ Adjust confidence threshold based on use case
- ✅ Use appropriate image size
- ✅ Batch process for efficiency
- ✅ Cache model for multiple predictions
- ✅ Monitor inference time

### Deployment
- ✅ Export to optimized format (ONNX, TFLite)
- ✅ Quantize model for mobile
- ✅ Use smaller model (yolov8n) for edge devices
- ✅ Implement proper error handling
- ✅ Add logging for monitoring

## 🔮 Future Improvements

- [ ] Video stream detection
- [ ] Real-time webcam detection
- [ ] Mobile app deployment (TFLite)
- [ ] Edge device deployment (Jetson Nano)
- [ ] Multi-scale detection
- [ ] Tracking implementation
- [ ] API service deployment
- [ ] Cloud deployment (AWS/GCP)
- [ ] Performance benchmarking
- [ ] Model ensemble

## 📚 Resources & References

### Official Documentation
- YOLOv8 Docs: https://docs.ultralytics.com/
- PyTorch: https://pytorch.org/docs/
- Ultralytics GitHub: https://github.com/ultralytics/ultralytics

### Papers
- YOLOv8: https://arxiv.org/abs/2305.09972
- Original YOLO: https://arxiv.org/abs/1506.02640

### Tutorials
- Ultralytics Tutorials: https://www.ultralytics.com/tutorials
- YOLOv8 Training Guide: https://docs.ultralytics.com/modes/train/

## 🤝 Contributing

Contributions are very welcome! To contribute:
1. Fork repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## 👨‍💻 Author

**Gus Agung Dev**
- Project: Traffic Sign Detection with YOLOv8
- Course: Data Science Programming - Semester 6
- Institution: [Your University]
- GitHub: [@gusagung](https://github.com/gusagung)

## 📞 Contact & Support

For questions, bug reports, or feature requests:
- Open an issue on GitHub
- Email: [your.email@example.com]

## 🙏 Acknowledgments

- Ultralytics team for YOLOv8
- Google Colab for free GPU access
- Dataset contributors
- DSP course lecturers and assistants
- Open-source community

## 📄 License

This project was created for academic purposes (Data Science Programming Course).

---

**Made with ❤️ by Gus Agung Dev**  
**Last Updated:** December 4, 2025  
**Version:** 1.0.0

🚀 **Happy Detecting!** 🚦
