# Traffic Sign Detection

Repository ini berisi implementasi sistem deteksi rambu lalu lintas menggunakan dua pendekatan berbeda: Convolutional Neural Network (CNN) dan YOLO (You Only Look Once).

## 📁 Struktur Project

### 1. CNN MODEL
Folder ini berisi implementasi deteksi rambu lalu lintas menggunakan pendekatan CNN klasik dengan berbagai arsitektur:
- **Model Architecture**: MobileNet, VGG16, dan Custom CNN
- **Dataset**: Dataset rambu lalu lintas Indonesia (cropped dan non-cropped)
- **Features**:
  - Training dan testing dengan berbagai konfigurasi
  - Dashboard untuk visualisasi hasil
  - Model dalam format `.h5` dan `.keras`
  - Class indices untuk mapping kategori rambu

**File Utama:**
- `DSP-CNN.ipynb` - Notebook training model CNN
- `Dashboard.py` - Dashboard untuk testing dan visualisasi
- `model_rambu_lalu_lintas.h5` - Model CNN terlatih

### 2. YOLO MODEL
Folder ini berisi implementasi deteksi rambu lalu lintas menggunakan YOLOv8:
- **Model**: YOLOv8n (Nano)
- **Dataset**: Format YOLO (images + labels dalam format txt)
- **Features**:
  - Real-time object detection
  - Multiple training experiments (train1-train1221)
  - Testing dan validasi
  - Dashboard terintegrasi

**File Utama:**
- `DSP-YOLO.ipynb` - Notebook training YOLOv8
- `Dashboard.py` - Dashboard untuk testing
- `DashboardGabung.py` - Dashboard gabungan CNN dan YOLO
- `data.yaml` - Konfigurasi dataset YOLO
- `runs/` - Hasil training dan deteksi

## 🚀 Cara Menggunakan

### CNN Model
1. Buka `CNN MODEL/DSP-CNN.ipynb` untuk training
2. Jalankan `Dashboard.py` untuk testing dengan model yang sudah dilatih
3. Model tersimpan di folder `MODEL/`

### YOLO Model
1. Buka `YOLO MODEL/DSP-YOLO.ipynb` untuk training YOLOv8
2. Gunakan `Dashboard.py` atau `DashboardGabung.py` untuk testing
3. Hasil training tersimpan di folder `runs/detect/`

## 📊 Dataset

Dataset berisi berbagai kategori rambu lalu lintas Indonesia:
- Lampu lalu lintas (hijau, kuning, merah)
- Rambu larangan (belok kanan/kiri, parkir, berhenti, dll)
- Rambu peringatan (zebra cross, isyarat lalu lintas, dll)
- Rambu petunjuk dan perintah

## 🛠️ Requirements

```
tensorflow
keras
ultralytics
opencv-python
numpy
pandas
matplotlib
streamlit (untuk dashboard)
```

## 📝 Notes

- File model (`.h5`, `.keras`, `.pt`) dan dataset berukuran besar mungkin tidak di-upload ke repository
- Untuk menggunakan model, download atau train ulang menggunakan notebook yang tersedia
- Dashboard dapat dijalankan menggunakan Streamlit

## 👥 Author

Ida Bagus Agung

## 📄 License

Educational Project - Data Science Programming
