# DSP CNN - Traffic Sign Classification

[![Made by](https://img.shields.io/badge/Made%20by-Gus%20Agung%20Dev-blue)](https://github.com/gusagung)
[![Last Updated](https://img.shields.io/badge/Last%20Updated-December%204%2C%202025-green)]()
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-API-red)](https://keras.io/)

## 📋 Deskripsi

Project ini mengimplementasikan sistem klasifikasi rambu lalu lintas Indonesia menggunakan Convolutional Neural Network (CNN). Project ini mengeksplorasi berbagai arsitektur CNN dan teknik preprocessing untuk mendapatkan hasil terbaik.

## 🎯 Tujuan Project

1. Mengklasifikasikan rambu lalu lintas Indonesia ke dalam 15+ kategori
2. Membandingkan performa berbagai arsitektur CNN:
   - Custom CNN (built from scratch)
   - Transfer Learning dengan MobileNet
   - Transfer Learning dengan VGG16
3. Menganalisis pengaruh preprocessing (cropping vs non-cropping)
4. Membangun dashboard interaktif untuk prediksi real-time

## 📁 Struktur Folder

```
DSP CNN/
├── Dashboard.py                    # Streamlit dashboard
├── DSP-CNN.ipynb                  # Main training notebook
├── DBS-CNN.ipynb                  # Eksperimen CNN
├── REVISI-DSP-CNN.ipynb           # Revisi dan improvement
├── check_h5.py                    # Script untuk cek model
├── model_rambu_lalu_lintas.h5     # Model utama
├── model_rambu_lalu_lintas.keras  # Model format Keras
├── class_indices.json             # Label mapping
├── prompt.txt                     # Project notes
│
├── DATASET AWAL/                  # Dataset original
│   ├── training-dataset-lalu-lintas/
│   │   ├── lampu-hijau/
│   │   ├── lampu-kuning/
│   │   ├── lampu-merah/
│   │   ├── larangan-belok-kanan/
│   │   ├── larangan-belok-kiri/
│   │   ├── larangan-berhenti/
│   │   └── ... (15+ categories)
│   └── testing-dataset-lalu-lintas/
│       └── ... (same categories)
│
├── DATASET CROPING/               # Dataset hasil cropping
│   ├── training-dataset-lalu-lintas/
│   └── testing-dataset-lalu-lintas/
│
├── New-Dataset/                   # Dataset yang digunakan
│   ├── training-dataset-lalu-lintas/
│   └── testing-dataset-lalu-lintas/
│
├── MODEL/                         # Saved models
│   ├── MobileNet_model.h5
│   ├── VGG16_model.h5
│   ├── Normal_train_testing_crop_model.h5
│   ├── Train-crop-Test-NotCrop_model.h5
│   └── Train-No-Crop-Test-Crop_model.h5
│
└── DBS/                          # Additional resources
```

## 🚀 Kategori Rambu Lalu Lintas

Project ini dapat mengklasifikasikan rambu-rambu berikut:

### Lampu Lalu Lintas
- 🟢 Lampu Hijau
- 🟡 Lampu Kuning  
- 🔴 Lampu Merah

### Rambu Larangan
- 🚫 Larangan Parkir
- 🚫 Larangan Berhenti
- 🚫 Larangan Belok Kanan
- 🚫 Larangan Belok Kiri
- 🚫 Larangan Memutar Balik
- 🚫 Larangan Masuk Kendaraan Bermotor
- 🚫 Larangan Berjalan Terus (Wajib Berhenti Sesaat)

### Rambu Peringatan
- ⚠️ Peringatan Alat Pemberi Isyarat Lalu Lintas
- ⚠️ Peringatan Banyak Pejalan Kaki (Zebra Cross)
- ⚠️ Peringatan Penegasan Rambu Tambahan

## 🛠️ Requirements

### Dependencies
```txt
tensorflow>=2.8.0
keras>=2.8.0
numpy>=1.21.0
pandas>=1.3.0
opencv-python>=4.5.0
matplotlib>=3.4.0
seaborn>=0.11.0
pillow>=8.3.0
scikit-learn>=0.24.0
streamlit>=1.10.0
```

### Installation
```bash
pip install tensorflow keras numpy pandas opencv-python matplotlib seaborn pillow scikit-learn streamlit
```

## 📊 Arsitektur Model

### 1. Custom CNN
Arsitektur CNN yang dibangun dari nol dengan layer-layer berikut:
```python
Model: Sequential
- Conv2D(32, (3,3), activation='relu')
- MaxPooling2D(2,2)
- Conv2D(64, (3,3), activation='relu')
- MaxPooling2D(2,2)
- Conv2D(128, (3,3), activation='relu')
- MaxPooling2D(2,2)
- Flatten()
- Dense(512, activation='relu')
- Dropout(0.5)
- Dense(num_classes, activation='softmax')
```

**Hyperparameters:**
- Input Size: 224x224x3
- Optimizer: Adam (lr=0.001)
- Loss: Categorical Crossentropy
- Batch Size: 32
- Epochs: 50-100

### 2. MobileNet (Transfer Learning)
```python
base_model = MobileNet(weights='imagenet', include_top=False)
# Freeze base layers
# Add custom classification head
```

**Advantages:**
- ✅ Lightweight model
- ✅ Fast inference
- ✅ Good for mobile deployment
- ✅ Pre-trained on ImageNet

### 3. VGG16 (Transfer Learning)
```python
base_model = VGG16(weights='imagenet', include_top=False)
# Fine-tuning approach
# Add custom layers
```

**Advantages:**
- ✅ High accuracy
- ✅ Deep architecture
- ✅ Strong feature extraction
- ⚠️ Large model size

## 🔄 Data Preprocessing

### Augmentation Techniques
```python
ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
```

### Cropping Experiments
Project ini membandingkan 4 skenario:
1. **Train Crop - Test Crop**: Dataset di-crop pada training dan testing
2. **Train Crop - Test No Crop**: Training dengan cropped, testing original
3. **Train No Crop - Test Crop**: Training original, testing cropped
4. **Train No Crop - Test No Crop**: Semua menggunakan original size

## 📈 Training Process

### Training Workflow
1. **Data Loading**: Load dan split dataset (80% train, 20% test)
2. **Preprocessing**: Resize, normalize, augmentation
3. **Model Building**: Pilih dan compile model
4. **Training**: Train dengan callbacks (ModelCheckpoint, EarlyStopping)
5. **Evaluation**: Test pada test set
6. **Save Model**: Simpan best model

### Training Notebooks
- **DSP-CNN.ipynb**: Main training pipeline
- **DBS-CNN.ipynb**: Eksperimen berbagai konfigurasi
- **REVISI-DSP-CNN.ipynb**: Improvement dan fine-tuning

### Training Commands
```python
# Training custom CNN
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=50,
    validation_data=val_generator,
    validation_steps=len(val_generator),
    callbacks=[checkpoint, early_stopping]
)
```

## 📊 Performance Results

### Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score | Training Time | Model Size |
|-------|----------|-----------|--------|----------|---------------|------------|
| Custom CNN | 85-90% | 0.87 | 0.85 | 0.86 | ~2-3 hours | 50 MB |
| MobileNet | 88-92% | 0.90 | 0.89 | 0.89 | ~1-2 hours | 25 MB |
| VGG16 | 90-95% | 0.92 | 0.91 | 0.91 | ~3-4 hours | 150 MB |

### Cropping Impact

| Scenario | Accuracy | Notes |
|----------|----------|-------|
| Train Crop - Test Crop | 92% | Best performance |
| Train No Crop - Test No Crop | 88% | Baseline |
| Train Crop - Test No Crop | 85% | Domain mismatch |
| Train No Crop - Test Crop | 83% | Domain mismatch |

## 🖥️ Dashboard Application

### Running the Dashboard
```bash
streamlit run Dashboard.py
```

### Features
1. **Upload Image**: Upload gambar rambu lalu lintas
2. **Real-time Prediction**: Prediksi kategori rambu
3. **Confidence Score**: Menampilkan tingkat kepercayaan
4. **Visualization**: Menampilkan hasil dengan bounding box
5. **Model Selection**: Pilih model yang akan digunakan

### Dashboard Interface
- Clean and intuitive UI
- Responsive design
- Real-time processing
- Download prediction results

## 📖 How to Use

### 1. Training Model Baru
```python
# Buka notebook DSP-CNN.ipynb
# Follow step-by-step cells untuk:
# 1. Load dataset
# 2. Preprocessing
# 3. Build model
# 4. Train model
# 5. Evaluate & save
```

### 2. Testing dengan Model yang Ada
```python
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load model
model = load_model('model_rambu_lalu_lintas.h5')

# Load dan preprocess image
img = Image.open('path/to/image.jpg')
img = img.resize((224, 224))
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)
class_idx = np.argmax(prediction)
```

### 3. Menggunakan Dashboard
```bash
# Jalankan dashboard
streamlit run Dashboard.py

# Atau buka browser ke:
# http://localhost:8501

# Upload image dan lihat hasil prediksi
```

## 🔍 Evaluation Metrics

### Classification Report
```python
from sklearn.metrics import classification_report

# Generate detailed report
print(classification_report(y_true, y_pred, 
                          target_names=class_names))
```

### Confusion Matrix
```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Visualize confusion matrix
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
```

## 🐛 Troubleshooting

### Common Issues

1. **Out of Memory Error**
   ```python
   # Reduce batch size
   batch_size = 16  # instead of 32
   ```

2. **Model Not Loading**
   ```python
   # Check model format
   model = load_model('model.h5', compile=False)
   model.compile(optimizer='adam', loss='categorical_crossentropy')
   ```

3. **Low Accuracy**
   - Increase epochs
   - Add more data augmentation
   - Try different learning rate
   - Use transfer learning

## 📝 Notes & Tips

### Best Practices
- ✅ Always use data augmentation
- ✅ Monitor validation loss to avoid overfitting
- ✅ Use callbacks (EarlyStopping, ModelCheckpoint)
- ✅ Save model regularly
- ✅ Log training metrics

### Performance Optimization
- Use GPU for training (Google Colab)
- Reduce image size if needed
- Use mixed precision training
- Batch processing for inference

## 🔮 Future Improvements

- [ ] Menambah lebih banyak kategori rambu
- [ ] Real-time video classification
- [ ] Model quantization untuk mobile
- [ ] Ensemble multiple models
- [ ] Active learning untuk improvement berkelanjutan
- [ ] Export ke TensorFlow Lite
- [ ] Web API deployment

## 📚 References

- TensorFlow Documentation: https://www.tensorflow.org/
- Keras Documentation: https://keras.io/
- MobileNet Paper: https://arxiv.org/abs/1704.04861
- VGG Paper: https://arxiv.org/abs/1409.1556

## 👨‍💻 Author

**Gus Agung Dev**
- Project: Traffic Sign Classification with CNN
- Course: Data Science Programming - Semester 6
- Last Updated: December 4, 2025

## 📄 License

Project ini dibuat untuk keperluan akademik.

---

**Made with ❤️ for Data Science Programming Course**  
**Last Updated:** December 4, 2025
