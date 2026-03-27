# SCT_ML_03
# 🐱🐶 SVM-Based Cats vs Dogs Classifier

This project builds an **image classification model** using **Support Vector Machine (SVM)** to classify images as **Cats or Dogs**.

---

## 📌 Project Overview

This project uses:

* Image preprocessing with OpenCV
* Feature extraction using **HOG (Histogram of Oriented Gradients)**
* Dimensionality reduction using **PCA**
* Classification using **SVM (Support Vector Machine)**

The model is trained on labeled image data and evaluated using multiple performance metrics.

---

## 📂 Dataset Structure

```id="ds1"
dataset/
└── train/
    ├── cat/
    │   ├── cat1.jpg
    │   ├── cat2.jpg
    ├── dog/
    │   ├── dog1.jpg
    │   ├── dog2.jpg
```

---

## ⚙️ Key Features

* HOG feature extraction for better image representation
* PCA for dimensionality reduction
* SVM with RBF kernel
* Model evaluation using:

  * Accuracy
  * Confusion Matrix
  * ROC Curve
  * Cross-validation

---

## 🧠 ML Pipeline

1. Load images
2. Resize to 64×64
3. Extract features (HOG / raw pixels)
4. Scale features
5. Apply PCA
6. Train SVM classifier

---

## 📊 Outputs

* ✅ Classification Report
* ✅ Confusion Matrix (`confusion_matrix.png`)
* ✅ ROC Curve (`roc_curve.png`)
* ✅ Sample Predictions (`sample_predictions.png`)
* ✅ Trained Model (`svm_model.pkl`)

---

## 🚀 How to Run

1. Update dataset path in code:

```id="run1"
TRAIN_DIR = "path_to_dataset/train"
```

2. Install dependencies:

```id="run2"
pip install -r requirements.txt
```

3. Run the script:

```id="run3"
python SCT_TASK_03.py
```

---

## 🔮 Prediction on New Image

```id="pred1"
from SCT_TASK_03 import predict_single_image
import joblib

model = joblib.load("svm_model.pkl")
predict_single_image("test.jpg", model)
```

---

## ⚙️ Configurable Parameters

* `IMG_SIZE` → Image size (default: 64×64)
* `MAX_SAMPLES` → Limit dataset size
* `USE_HOG` → Enable/disable HOG features
* `N_PCA_COMPONENTS` → PCA components
* `TUNE_HYPERPARAMS` → Enable GridSearch

---

## 💡 Use Cases

* Image classification systems
* Pet recognition apps
* Computer vision learning projects

---

## ⚠️ Notes

* Ensure dataset folder structure is correct
* Unicode paths are supported using a custom image loader
* DB performance depends on dataset size

---

## 📌 Future Improvements

* Convert into a **Streamlit web app**
* Use **CNN (Deep Learning)** for better accuracy
* Add real-time webcam prediction

---

## 👨‍💻 Author

Prashanth B
