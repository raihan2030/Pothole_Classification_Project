# Pothole Classification Project 🚗

A comprehensive machine learning project for detecting and classifying road surface conditions (potholes vs smooth roads) using multiple deep learning and traditional ML approaches.

## 📋 Project Overview

This project implements and compares various machine learning algorithms to classify road images into two categories:

- **berlubang** (Pothole/Damaged)
- **mulus** (Smooth/Normal)

The goal is to develop an accurate and robust classification system that can be used for road condition monitoring and maintenance planning.

## 🎯 Problem Statement

Road pothole detection is crucial for:

- Infrastructure maintenance planning
- Vehicle safety enhancement
- Cost-effective road management
- Automated inspection systems

This project explores multiple classification approaches, from traditional machine learning (HOG + SVM) to modern deep learning architectures (MobileNet, EfficientNetB0).

## 📊 Dataset Structure

```
dataset/
├── 2_processed/
│   ├── train/
│   │   ├── berlubang/  (Pothole images)
│   │   └── mulus/      (Smooth road images)
│   └── test/
│       ├── berlubang/
│       └── mulus/
```

**Dataset Characteristics:**

- Binary classification task
- Preprocessed and normalized images
- Balanced split between train, validation, and test sets
- 224×224 pixel images (for deep learning models)

## 🔬 Implemented Approaches

### 1. **Traditional Machine Learning**

#### 02_HOG_SVM.ipynb

- **Algorithm**: Histogram of Oriented Gradients (HOG) + Support Vector Machine (SVM)
- **Features**: Handcrafted HOG descriptors
- **Advantages**: Fast training, interpretable features, good baseline
- **Use Case**: Quick baseline comparison

#### 03_Hybrid_CNN_SVM.ipynb

- **Algorithm**: CNN feature extraction + SVM classification
- **Architecture**: Convolutional layers for feature learning + SVM for classification
- **Advantages**: Combines deep learning features with traditional ML
- **Use Case**: Bridge between classical and modern approaches

#### 04_MLP.ipynb

- **Algorithm**: Multi-Layer Perceptron (Fully Connected Neural Network)
- **Architecture**: Input → Dense → Dense → Output
- **Advantages**: Simple, easy to implement
- **Use Case**: Baseline neural network performance

#### 05_PNN.ipynb

- **Algorithm**: Probabilistic Neural Network
- **Architecture**: Pattern layer with Gaussian activation
- **Advantages**: Fast training, probabilistic outputs
- **Use Case**: Probabilistic classification baseline

### 2. **Deep Learning Approaches**

#### 06_DeepLearning_MobileNet.ipynb

- **Architecture**: MobileNetV2 (Pre-trained on ImageNet)
- **Key Features**:
  - Depthwise separable convolutions
  - Lightweight and mobile-friendly
  - Fast inference time
- **Training Strategy**:
  - Phase 1: Train custom head with frozen base
  - Phase 2: Fine-tune top layers
- **Best Model**: `mobilenet_best.keras`

#### 07_DeepLearning_EfficientNetB0.ipynb

- **Architecture**: EfficientNetB0 (Pre-trained on ImageNet)
- **Key Features**:
  - Compound scaling (depth, width, resolution)
  - Better accuracy-efficiency trade-off
  - State-of-the-art performance
- **Training Strategy**:
  - Phase 1: Custom head training (15 epochs)
  - Phase 2: Fine-tuning top 20 layers (30 epochs)
- **Best Model**: `efficientnet_best.keras`

## 🏗️ Project Structure

```
Pothole_Classification_Project/
├── notebooks/
│   ├── 01_Preprocessing.ipynb           # Data loading & preprocessing
│   ├── 02_HOG_SVM.ipynb                 # Traditional ML approach
│   ├── 03_Hybrid_CNN_SVM.ipynb          # Hybrid approach
│   ├── 04_MLP.ipynb                     # Simple neural network
│   ├── 05_PNN.ipynb                     # Probabilistic neural network
│   ├── 06_DeepLearning_MobileNet.ipynb  # MobileNetV2
│   └── 07_DeepLearning_EfficientNetB0.ipynb  # EfficientNetB0
├── saved_models/
│   ├── efficientnet_best.keras          # Best EfficientNetB0 model
│   ├── efficientnet_model.keras         # Final EfficientNetB0 model
│   ├── mobilenet_best.keras             # Best MobileNetV2 model
│   ├── mobilenet_model.keras            # Final MobileNetV2 model
│   └── mobilenet_feature_extractor.keras # Feature extractor
├── outputs/
│   ├── training_history_*.png           # Training curves
│   ├── cm_*.png                         # Confusion matrices
│   ├── grad_cam_*.png                   # Grad-CAM visualizations
│   ├── cv_scores_*.png                  # Cross-validation plots
│   └── (other evaluation results)
├── dataset/
│   └── 2_processed/
│       ├── train/
│       │   ├── berlubang/
│       │   └── mulus/
│       └── test/
│           ├── berlubang/
│           └── mulus/
└── README.md
```

## 🔑 Key Features

### Data Preprocessing (01_Preprocessing.ipynb)

- Image loading and validation
- Normalization and resizing
- Data augmentation techniques
- Train-validation-test split

### Model Architecture Highlights

**EfficientNetB0 & MobileNetV2 Common Features:**

```python
- Input: 224×224×3 images
- Base: Pre-trained ImageNet weights (frozen)
- Custom Head:
  - GlobalAveragePooling2D
  - Dense(128, relu, L2 regularization)
  - Dropout(0.5)
  - Dense(64, relu, L2 regularization)
  - Dropout(0.3)
  - Dense(1, sigmoid) → Binary output
- Optimizer: Adam (adaptive learning rates)
- Loss: Binary Crossentropy
```

### Training Strategy

1. **Phase 1**: Custom head training (base model frozen)
   - Focus on learning task-specific features
   - Prevents catastrophic forgetting
2. **Phase 2**: Fine-tuning
   - Unfreeze top layers of base model
   - Lower learning rate to preserve learned features
   - Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

### Evaluation Metrics

For each model, the following metrics are computed:

- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Classification breakdown
- **Cross-Validation Score**: Robustness assessment (5-fold)

### Advanced Visualizations

#### Grad-CAM (Gradient-weighted Class Activation Mapping)

- Visualizes which regions of the image influence the model's decision
- Shows attention areas for both pothole and smooth roads
- Helps validate that the model learns meaningful features

#### Training History

- Loss curves (training vs validation)
- Accuracy curves with phase transition markers
- Helps identify overfitting/underfitting patterns

#### Cross-Validation Analysis

- Per-fold accuracy scores
- Mean and standard deviation
- Comparison with test accuracy
- Gap analysis for model stability

### Automated Diagnosis

Each deep learning notebook includes automatic model health diagnosis:

- **Underfitting Detection**: Both CV and test accuracy are low
- **Overfitting Detection**: Large gap between train and test performance
- **Distribution Issues**: CV and test accuracies diverge significantly
- **Variance Analysis**: High cross-fold variance indicates instability

## 🚀 Usage Instructions

### Prerequisites

```bash
# Required libraries
tensorflow
opencv-python
scikit-learn
matplotlib
seaborn
numpy
tqdm
```

### Running the Project

1. **Data Preprocessing**

   ```
   Open and run: 01_Preprocessing.ipynb
   ```

2. **Train Traditional ML Models**

   ```
   Open and run:
   - 02_HOG_SVM.ipynb
   - 03_Hybrid_CNN_SVM.ipynb
   - 04_MLP.ipynb
   - 05_PNN.ipynb
   ```

3. **Train Deep Learning Models**

   ```
   Open and run:
   - 06_DeepLearning_MobileNet.ipynb
   - 07_DeepLearning_EfficientNetB0.ipynb
   ```

4. **Evaluate Results**
   - Check `outputs/` directory for visualizations
   - Compare metrics across all models
   - Review Grad-CAM outputs for interpretability

### Loading Trained Models

```python
from tensorflow.keras.models import load_model

# Load EfficientNetB0
model = load_model('saved_models/efficientnet_best.keras')

# Make predictions
predictions = model.predict(image_array)
```

## 📈 Expected Results Summary

### Performance Comparison Framework

- **Traditional ML**: Fast, interpretable, baseline performance
- **Hybrid Approach**: Balanced feature learning and classification
- **Simple NN**: Basic deep learning baseline
- **MobileNetV2**: Lightweight, good real-time performance
- **EfficientNetB0**: Superior accuracy-efficiency trade-off, recommended for production

### Key Evaluation Criteria

1. **Accuracy**: Overall classification correctness
2. **Robustness**: Cross-validation score stability
3. **Generalization**: Test accuracy vs CV score gap
4. **Interpretability**: Grad-CAM visualization quality
5. **Efficiency**: Model size and inference time

## 🔍 Model Interpretability

The project includes several interpretability features:

1. **Grad-CAM Heatmaps**: Shows decision-making regions
2. **Feature Extraction**: Visualizes learned patterns
3. **Classification Reports**: Detailed per-class metrics
4. **Confusion Matrices**: Misclassification patterns
5. **Cross-Validation Analysis**: Model stability assessment

## 💡 Key Insights & Best Practices

### Why Two-Phase Training?

- **Phase 1**: Adapts pre-trained features to pothole classification task
- **Phase 2**: Fine-tunes general features for better performance
- **Result**: Faster convergence and better final accuracy

### Regularization Techniques Used

- **L2 Regularization**: Prevents weight explosion
- **Dropout Layers**: Reduces co-adaptation of neurons
- **Early Stopping**: Prevents overfitting
- **Learning Rate Reduction**: Fine-tunes convergence

### Why EfficientNetB0?

- Optimal scaling of depth, width, and resolution
- Better performance-to-parameter ratio than standard CNNs
- Pre-trained ImageNet weights provide strong transfer learning foundation
- Suitable for mobile and edge deployment

## 📚 Technologies & Libraries

- **Deep Learning**: TensorFlow, Keras
- **Computer Vision**: OpenCV
- **Machine Learning**: Scikit-learn
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn
- **Progress Tracking**: tqdm

## 🎓 Learning Outcomes

This project demonstrates:

1. Transfer learning with pre-trained models
2. Fine-tuning strategies for domain-specific tasks
3. Hyperparameter optimization
4. Model evaluation and diagnosis
5. Binary image classification pipeline
6. Comparison of classical vs deep learning approaches
7. Model interpretability techniques

## 📝 Notes

- All notebooks are configured to run in Google Colab (with drive mounting)
- Models are saved in Keras format (.keras) for compatibility
- All visualizations are saved to the `outputs/` directory
- Cross-validation uses Stratified K-Fold for balanced splits

## 🔗 References

- EfficientNet: [Tan & Le, 2019](https://arxiv.org/abs/1905.11946)
- MobileNetV2: [Sandler et al., 2018](https://arxiv.org/abs/1801.04381)
- Grad-CAM: [Selvaraju et al., 2016](https://arxiv.org/abs/1610.02055)

## 📧 Project Information

**Project Type**: Machine Learning Classification  
**Task**: Binary Image Classification (Pothole Detection)  
**Duration**: University Course (UAS - Ujian Akhir Semester)  
**Semester**: 6  
**Course**: Pembelajaran Mesin (Machine Learning)

---

**Status**: ✅ Complete with multiple approaches and comprehensive evaluation

_Last Updated: 2026_
