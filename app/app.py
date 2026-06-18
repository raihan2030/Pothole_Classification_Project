import streamlit as st
import numpy as np
import cv2
import joblib
from skimage.feature import hog
from PIL import Image
from pathlib import Path
import os

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_input_mobilenet
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_input_effnet
from tensorflow.keras.models import Model, load_model

# 1. KONFIGURASI PATH & PARAMETER 
CURRENT_DIR = Path(os.path.dirname(__file__))
MODEL_DIR = CURRENT_DIR.parent / "saved_models"

CLASSES = ["berlubang", "mulus"]

# Parameter HOG 
HOG_SIZE = (64, 64) 
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_BLOCK_NORM = "L2-Hys"

# Parameter CNN (Untuk Hybrid, MobileNet, & EfficientNet)
CNN_IMG_SIZE = (224, 224)

# Parameter MLP
MLP_IMG_SIZE = (64, 64)

# Parameter PNN
PNN_IMG_SIZE = (32, 32)

# 2. DEFINISI KELAS KUSTOM PNN (WAJIB ADA)
class PNN:
    def __init__(self, std=0.1, batch_size=10):
        self.std        = std
        self.batch_size = batch_size
        self.X_train    = None
        self.y_train    = None
        self.classes_   = None

    def fit(self, X, y):
        self.X_train  = X
        self.y_train  = y
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        proba = np.zeros((len(X), len(self.classes_)))
        for batch_start in range(0, len(X), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(X))
            X_batch   = X[batch_start:batch_end]
            
            dist_sq_all = []
            for i, cls in enumerate(self.classes_):
                X_cls   = self.X_train[self.y_train == cls]
                diff    = X_batch[:, np.newaxis, :] - X_cls[np.newaxis, :, :]
                dist_sq = np.sum(diff ** 2, axis=2)
                dist_sq_all.append(dist_sq)
                
            min_dist_sq = np.zeros((len(X_batch), 1))
            for b in range(len(X_batch)):
                min_d = np.inf
                for dist_sq in dist_sq_all:
                    min_d = min(min_d, np.min(dist_sq[b]))
                min_dist_sq[b, 0] = min_d

            for i, cls in enumerate(self.classes_):
                dist_sq = dist_sq_all[i]
                shifted_dist_sq = dist_sq - min_dist_sq
                kernels = np.exp(-shifted_dist_sq / (2 * self.std ** 2))
                proba[batch_start:batch_end, i] = kernels.mean(axis=1)

        proba_sum = proba.sum(axis=1, keepdims=True)
        proba_sum = np.where(proba_sum == 0, 1, proba_sum)
        return proba / proba_sum

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

# 3. FUNGSI LOAD MODEL & FEATURE EXTRACTOR
@st.cache_resource
def load_hog_svm_model():
    try:
        return joblib.load(MODEL_DIR / "svm_hog_model.pkl"), joblib.load(MODEL_DIR / "scaler_hog.pkl")
    except Exception as e:
        st.error(f"Gagal memuat model HOG SVM: {e}")
        return None, None

@st.cache_resource
def load_cnn_svm_model():
    try:
        return joblib.load(MODEL_DIR / "svm_cnn_model.pkl"), joblib.load(MODEL_DIR / "scaler_cnn.pkl")
    except Exception as e:
        st.error(f"Gagal memuat model Hybrid CNN SVM: {e}")
        return None, None

@st.cache_resource
def load_cnn_feature_extractor():
    base_model = MobileNetV2(input_shape=(*CNN_IMG_SIZE, 3), include_top=False, weights="imagenet")
    base_model.trainable = False
    return Model(inputs=base_model.input, outputs=tf.keras.layers.GlobalAveragePooling2D()(base_model.output))

@st.cache_resource
def load_mobilenet_model():
    try:
        model_path = MODEL_DIR / "mobilenet_model.keras"
        if not model_path.exists():
            model_path = MODEL_DIR / "mobilenet_model.h5"
        return load_model(model_path)
    except Exception as e:
        st.error(f"Gagal memuat model MobileNet: {e}")
        return None

@st.cache_resource
def load_efficientnet_model():
    try:
        model_path = MODEL_DIR / "efficientnet_model.keras"
        if not model_path.exists():
            model_path = MODEL_DIR / "efficientnet_model.h5"
        return load_model(model_path)
    except Exception as e:
        st.error(f"Gagal memuat model EfficientNet: {e}")
        return None

@st.cache_resource
def load_mlp_model():
    try:
        return joblib.load(MODEL_DIR / "mlp_model.pkl"), joblib.load(MODEL_DIR / "scaler_mlp.pkl")
    except Exception as e:
        st.error(f"Gagal memuat model MLP: {e}")
        return None, None

@st.cache_resource
def load_pnn_model():
    try:
        return joblib.load(MODEL_DIR / "pnn_model.pkl"), joblib.load(MODEL_DIR / "scaler_pnn.pkl")
    except Exception as e:
        st.error(f"Gagal memuat model PNN: {e}")
        return None, None


# 4. FUNGSI EKSTRAKSI & PREDIKSI
def predict_hog_svm(image_cv, svm_model, scaler):
    img_gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, HOG_SIZE, interpolation=cv2.INTER_AREA)
    features = hog(img_resized, orientations=HOG_ORIENTATIONS, pixels_per_cell=HOG_PIXELS_PER_CELL,
                cells_per_block=HOG_CELLS_PER_BLOCK, block_norm=HOG_BLOCK_NORM, feature_vector=True)
    features_scaled = scaler.transform(features.reshape(1, -1))
    
    prediction_idx = svm_model.predict(features_scaled)[0]
    confidence = svm_model.predict_proba(features_scaled)[0][prediction_idx] * 100
    return CLASSES[prediction_idx], confidence

def predict_cnn_svm(image_cv, feature_extractor, svm_model, scaler):
    img_resized = cv2.resize(image_cv, CNN_IMG_SIZE, interpolation=cv2.INTER_AREA)
    img_pre = preprocess_input_mobilenet(img_resized.astype(np.float32))
    img_batch = np.expand_dims(img_pre, axis=0)
    
    features = feature_extractor.predict(img_batch, verbose=0)
    features_scaled = scaler.transform(features)
    
    prediction_idx = svm_model.predict(features_scaled)[0]
    confidence = svm_model.predict_proba(features_scaled)[0][prediction_idx] * 100
    return CLASSES[prediction_idx], confidence

def predict_mobilenet(image_cv, model):
    img_resized = cv2.resize(image_cv, CNN_IMG_SIZE, interpolation=cv2.INTER_AREA)
    img_pre = preprocess_input_mobilenet(img_resized.astype(np.float32))
    img_batch = np.expand_dims(img_pre, axis=0)
    
    pred_prob = model.predict(img_batch, verbose=0)[0][0]
    if pred_prob >= 0.5:
        return CLASSES[1], pred_prob * 100
    else:
        return CLASSES[0], (1.0 - pred_prob) * 100

def predict_efficientnet(image_cv, model):
    img_resized = cv2.resize(image_cv, CNN_IMG_SIZE, interpolation=cv2.INTER_AREA)
    # Gunakan preprocess khusus EfficientNet
    img_pre = preprocess_input_effnet(img_resized.astype(np.float32))
    img_batch = np.expand_dims(img_pre, axis=0)
    
    pred_prob = model.predict(img_batch, verbose=0)[0][0]
    if pred_prob >= 0.5:
        return CLASSES[1], pred_prob * 100
    else:
        return CLASSES[0], (1.0 - pred_prob) * 100

def predict_mlp(image_cv, mlp_model, scaler):
    img_resized = cv2.resize(image_cv, MLP_IMG_SIZE, interpolation=cv2.INTER_AREA)
    img_norm = img_resized.astype(np.float32) / 255.0
    features_scaled = scaler.transform(img_norm.flatten().reshape(1, -1))
    
    prediction_idx = mlp_model.predict(features_scaled)[0]
    confidence = mlp_model.predict_proba(features_scaled)[0][prediction_idx] * 100
    return CLASSES[prediction_idx], confidence

def predict_pnn(image_cv, pnn_model, scaler):
    img_resized = cv2.resize(image_cv, PNN_IMG_SIZE, interpolation=cv2.INTER_AREA)
    img_norm = img_resized.astype(np.float32) / 255.0
    features_scaled = scaler.transform(img_norm.flatten().reshape(1, -1))
    
    proba = pnn_model.predict_proba(features_scaled)[0]
    prediction_idx = np.argmax(proba)
    confidence = proba[prediction_idx] * 100
    return CLASSES[prediction_idx], confidence


# 5. ANTARMUKA STREAMLIT UTAMA
st.set_page_config(page_title="Pothole Classification", page_icon="🛣️", layout="wide")

# Preload feature extractor Hybrid CNN
feature_extractor = load_cnn_feature_extractor()

st.sidebar.title("Pilihan Model")
st.sidebar.write("Silakan pilih arsitektur model yang ingin diuji:")

selected_model = st.sidebar.radio(
    "Daftar Model:",
    (
        "1. HOG + SVM", 
        "2. Hybrid CNN + SVM", 
        "3. MLP Classifier", 
        "4. PNN Classifier", 
        "5. MobileNetV2 (End-to-End)", 
        "6. EfficientNetB0 (End-to-End)"
    )
)

st.title("Klasifikasi Kondisi Jalan (Berlubang vs Mulus)")
st.write("Unggah gambar permukaan jalan untuk dianalisis oleh sistem.")

uploaded_file = st.file_uploader("Pilih file gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file)
    
    image_cv = np.array(image_pil)
    if len(image_cv.shape) == 2:
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_GRAY2RGB)
    elif image_cv.shape[2] == 4:
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGBA2RGB)
    elif len(image_cv.shape) == 3 and selected_model == "1. HOG + SVM":
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
        
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image_pil, caption="Gambar yang Diunggah", use_container_width=True)

    with col2:
        st.markdown("### Hasil Analisis")
        
        if selected_model == "1. HOG + SVM":
            st.info("Model Aktif: **HOG + SVM**")
            svm_model, scaler = load_hog_svm_model()
            if st.button("Jalankan Prediksi", use_container_width=True, type="primary"):
                if svm_model and scaler:
                    with st.spinner("Mengekstraksi fitur HOG & memprediksi..."):
                        label, conf = predict_hog_svm(image_cv, svm_model, scaler)
                        st.success("Analisis Selesai!")
                        st.metric(label="Status Jalan", value=label.upper())
                        st.metric(label="Tingkat Keyakinan", value=f"{conf:.2f}%")
                        
        elif selected_model == "2. Hybrid CNN + SVM":
            st.info("Model Aktif: **Hybrid MobileNetV2 + SVM**")
            svm_model, scaler = load_cnn_svm_model()
            if st.button("Jalankan Prediksi", use_container_width=True, type="primary"):
                if svm_model and scaler and feature_extractor:
                    with st.spinner("Ekstraksi fitur CNN & memprediksi..."):
                        label, conf = predict_cnn_svm(image_cv, feature_extractor, svm_model, scaler)
                        st.success("Analisis Selesai!")
                        st.metric(label="Status Jalan", value=label.upper())
                        st.metric(label="Tingkat Keyakinan", value=f"{conf:.2f}%")


        elif selected_model == "3. MLP Classifier":
            st.info("Model Aktif: **Multi-Layer Perceptron (MLP)**")
            mlp_model, scaler = load_mlp_model()
            if st.button("Jalankan Prediksi", use_container_width=True, type="primary"):
                if mlp_model and scaler:
                    with st.spinner("Memproses gambar & memprediksi dengan JST..."):
                        label, conf = predict_mlp(image_cv, mlp_model, scaler)
                        st.success("Analisis Selesai!")
                        st.metric(label="Status Jalan", value=label.upper())
                        st.metric(label="Tingkat Keyakinan", value=f"{conf:.2f}%")
                        
        elif selected_model == "4. PNN Classifier":
            st.info("Model Aktif: **Probabilistic Neural Network (PNN)**")
            pnn_model, scaler = load_pnn_model()
            if st.button("Jalankan Prediksi", use_container_width=True, type="primary"):
                if pnn_model and scaler:
                    with st.spinner("Membandingkan matriks gambar dengan neuron PNN..."):
                        label, conf = predict_pnn(image_cv, pnn_model, scaler)
                        st.success("Analisis Selesai!")
                        st.metric(label="Status Jalan", value=label.upper())
                        st.metric(label="Tingkat Keyakinan", value=f"{conf:.2f}%")

        elif selected_model == "5. MobileNetV2 (End-to-End)":
            st.info("Model Aktif: **Deep Learning MobileNetV2**")
            mobilenet_model = load_mobilenet_model()
            if st.button("Jalankan Prediksi", use_container_width=True, type="primary"):
                if mobilenet_model:
                    with st.spinner("Memproses gambar & memprediksi..."):
                        label, conf = predict_mobilenet(image_cv, mobilenet_model)
                        st.success("Analisis Selesai!")
                        st.metric(label="Status Jalan", value=label.upper())
                        st.metric(label="Tingkat Keyakinan", value=f"{conf:.2f}%")

        elif selected_model == "6. EfficientNetB0 (End-to-End)":
            st.info("Model Aktif: **Deep Learning EfficientNetB0**")
            efficientnet_model = load_efficientnet_model()
            if st.button("Jalankan Prediksi", use_container_width=True, type="primary"):
                if efficientnet_model:
                    with st.spinner("Menganalisis gambar menggunakan EfficientNetB0..."):
                        label, conf = predict_efficientnet(image_cv, efficientnet_model)
                        st.success("Analisis Selesai!")
                        st.metric(label="Status Jalan", value=label.upper())
                        st.metric(label="Tingkat Keyakinan", value=f"{conf:.2f}%")