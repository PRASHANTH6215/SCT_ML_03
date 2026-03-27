import streamlit as st
import cv2
import numpy as np
import joblib
from skimage.feature import hog


model = joblib.load(r"C:\Users\PRASHANTH\OneDrive\文档\Desktop\SCT_TASK_03\svm_model.pkl")
st.title("🐶🐱 Cats vs Dogs Classifier (SVM + HOG)")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="Uploaded Image", use_column_width=True)

    img = cv2.resize(img, (64, 64))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    features = hog(
        gray,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        feature_vector=True
    )

    features = features.reshape(1, -1)

    prediction = model.predict(features)

    if prediction[0] == 0:
        st.success("🐱 It's a CAT!")
    else:
        st.success("🐶 It's a DOG!")