
import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from PIL import Image
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import sobel


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "svm_pneumonia_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "feature_scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "selected_features.json")


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

with open(FEATURES_PATH, "r") as f:
    selected_features = json.load(f)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_prediction_features(image):

    # Convert to grayscale
    image = image.convert("L")

    # Resize to PneumoniaMNIST size
    image = image.resize((28, 28))

    # Normalize exactly as used during training
    img = np.asarray(image, dtype=np.float32) / 255.0

    features = {}

    # Intensity features
    features["mean_intensity"] = np.mean(img)
    features["std_intensity"] = np.std(img)
    features["min_intensity"] = np.min(img)
    features["max_intensity"] = np.max(img)
    features["median_intensity"] = np.median(img)

    # Percentiles
    features["percentile_10"] = np.percentile(img, 10)
    features["percentile_25"] = np.percentile(img, 25)
    features["percentile_75"] = np.percentile(img, 75)
    features["percentile_90"] = np.percentile(img, 90)

    # Entropy
    hist, _ = np.histogram(img, bins=256, range=(0, 1))
    prob = hist / np.sum(hist)
    prob = prob[prob > 0]

    features["entropy"] = -np.sum(
        prob * np.log2(prob)
    )

    # ========================================================
    # GLCM FEATURES
    # ========================================================

    img_uint8 = (img * 255).astype(np.uint8)

    glcm = graycomatrix(
        img_uint8,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )

    features["glcm_contrast"] = graycoprops(
        glcm, "contrast"
    )[0, 0]

    features["glcm_dissimilarity"] = graycoprops(
        glcm, "dissimilarity"
    )[0, 0]

    features["glcm_homogeneity"] = graycoprops(
        glcm, "homogeneity"
    )[0, 0]

    features["glcm_energy"] = graycoprops(
        glcm, "energy"
    )[0, 0]

    features["glcm_correlation"] = graycoprops(
        glcm, "correlation"
    )[0, 0]

    # ========================================================
    # EDGE DENSITY
    # ========================================================

    edges = sobel(img)

    threshold = np.mean(edges)

    features["edge_density"] = np.mean(
        edges > threshold
    )

    return features


# ============================================================
# PREDICTION
# ============================================================

def predict_pneumonia(image):

    if image is None:
        return None, "Please upload a chest X-ray image."

    # Extract features
    features = extract_prediction_features(image)

    # Exact feature order used during training
    X = pd.DataFrame(
        [[features[f] for f in selected_features]],
        columns=selected_features
    )

    # Scale
    X_scaled = scaler.transform(X.values)

    # Prediction
    prediction = model.predict(X_scaled)[0]

    # Probability
    probabilities = model.predict_proba(X_scaled)[0]

    normal_probability = probabilities[0] * 100
    pneumonia_probability = probabilities[1] * 100

    # Result
    if prediction == 1:

        result = "### 🔴 PNEUMONIA"

        recommendation = (
            "The model classified this image as pneumonia-positive.\n\n"
            "**Recommendation:** Further clinical and radiological "
            "evaluation is recommended."
        )

    else:

        result = "### 🟢 NORMAL"

        recommendation = (
            "The model classified this image as normal.\n\n"
            "**Recommendation:** Clinical assessment should still be "
            "considered when symptoms are present."
        )

    probability_text = (
        f"**Normal probability:** {normal_probability:.2f}%\n\n"
        f"**Pneumonia probability:** {pneumonia_probability:.2f}%"
    )

    return result, probability_text, recommendation


# ============================================================
# STREAMLIT USER INTERFACE
# ============================================================

st.set_page_config(
    page_title="Pneumonia Screening System",
    page_icon="🫁",
    layout="wide"
)

st.title("🫁 Pneumonia Screening System")

st.write(
    "Computer-aided pneumonia screening from chest X-ray images "
    "using handcrafted image features and an SVM classifier."
)

st.warning(
    "⚠️ This system is intended for research and screening support "
    "only. It is not a substitute for professional medical diagnosis."
)

st.divider()

uploaded_file = st.file_uploader(
    "Upload Chest X-ray",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Uploaded X-ray")

        st.image(
            image,
            use_container_width=True
        )

    with col2:

        st.subheader("Screening Result")

        if st.button(
            "🔍 Analyze X-ray",
            type="primary"
        ):

            result, probability = predict_pneumonia(image)[0:2]

            # Recalculate recommendation
            features = extract_prediction_features(image)

            X = pd.DataFrame(
                [[features[f] for f in selected_features]],
                columns=selected_features
            )

            X_scaled = scaler.transform(X.values)

            prediction = model.predict(X_scaled)[0]

            if prediction == 1:

                recommendation = (
                    "The model classified this image as "
                    "pneumonia-positive. Further clinical and "
                    "radiological evaluation is recommended."
                )

            else:

                recommendation = (
                    "The model classified this image as normal. "
                    "Clinical assessment should still be considered "
                    "when symptoms are present."
                )

            st.markdown(result)

            st.markdown("### Prediction Probabilities")

            st.info(probability)

            st.markdown(
                "### Clinical Support Recommendation"
            )

            st.write(recommendation)

st.divider()

st.caption(
    "Pneumonia Screening System | SVM-based research prototype"
)
