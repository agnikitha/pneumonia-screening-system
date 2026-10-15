
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
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Pneumonia Screening System",
    page_icon="🫁",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    with open(FEATURES_PATH, "r") as f:
        selected_features = json.load(f)

    if len(selected_features) != model.n_features_in_:
        raise ValueError(
            "The selected feature count does not match the model."
        )

    if scaler.n_features_in_ != model.n_features_in_:
        raise ValueError(
            "The scaler feature count does not match the model."
        )

    return model, scaler, selected_features


try:
    model, scaler, selected_features = load_model()
except Exception as e:
    st.error(f"Unable to load the screening model: {e}")
    st.stop()


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_prediction_features(image):

    # Convert to grayscale
    image = image.convert("L")

    # Resize to PneumoniaMNIST dimensions
    image = image.resize((28, 28))

    # Normalize pixel values
    img = np.asarray(image, dtype=np.float32) / 255.0

    features = {}

    # --------------------------------------------------------
    # INTENSITY FEATURES
    # --------------------------------------------------------

    features["mean_intensity"] = np.mean(img)
    features["std_intensity"] = np.std(img)
    features["min_intensity"] = np.min(img)
    features["max_intensity"] = np.max(img)
    features["median_intensity"] = np.median(img)

    # --------------------------------------------------------
    # PERCENTILE FEATURES
    # --------------------------------------------------------

    features["percentile_10"] = np.percentile(img, 10)
    features["percentile_25"] = np.percentile(img, 25)
    features["percentile_75"] = np.percentile(img, 75)
    features["percentile_90"] = np.percentile(img, 90)

    # --------------------------------------------------------
    # ENTROPY
    # --------------------------------------------------------

    hist, _ = np.histogram(
        img,
        bins=256,
        range=(0, 1)
    )

    prob = hist / np.sum(hist)
    prob = prob[prob > 0]

    features["entropy"] = -np.sum(
        prob * np.log2(prob)
    )

    # --------------------------------------------------------
    # GLCM FEATURES
    # IMPORTANT: MATCH THE TRAINING PIPELINE
    # --------------------------------------------------------

    img_uint8 = (img * 255).astype(np.uint8)

    # Quantize grayscale image to 32 levels
    img_quantized = (
        img_uint8.astype(np.uint16) * 32 // 256
    ).astype(np.uint8)

    glcm = graycomatrix(
        img_quantized,
        distances=[1],
        angles=[0],
        levels=32,
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

    # --------------------------------------------------------
    # EDGE DENSITY
    # --------------------------------------------------------

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

    # Extract features
    features = extract_prediction_features(image)

    # Arrange features in the exact model input order
    X = pd.DataFrame(
        [[features[name] for name in selected_features]],
        columns=selected_features
    )

    # Scale features
    X_scaled = scaler.transform(X)

    # Predict class
    prediction = int(model.predict(X_scaled)[0])

    # Predict class probabilities
    probabilities = model.predict_proba(X_scaled)[0]

    # Map probabilities to the corresponding model classes
    class_probabilities = dict(
        zip(model.classes_, probabilities)
    )

    normal_probability = (
        float(class_probabilities[0]) * 100
    )

    pneumonia_probability = (
        float(class_probabilities[1]) * 100
    )

    # --------------------------------------------------------
    # RESULT AND RECOMMENDATION
    # --------------------------------------------------------

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
            "**Recommendation:** Clinical assessment should still "
            "be considered when symptoms are present."
        )

    probability_text = (
        f"**Normal probability:** "
        f"{normal_probability:.2f}%\n\n"
        f"**Pneumonia probability:** "
        f"{pneumonia_probability:.2f}%"
    )

    return result, probability_text, recommendation


# ============================================================
# STREAMLIT USER INTERFACE
# ============================================================

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

    try:
        image = Image.open(uploaded_file).convert("RGB")

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

                with st.spinner("Analyzing image..."):

                    result, probability, recommendation = (
                        predict_pneumonia(image)
                    )

                st.markdown(result)

                st.markdown("### Prediction Probabilities")
                st.info(probability)

                st.markdown(
                    "### Clinical Support Recommendation"
                )

                st.write(recommendation)

    except Exception as e:
        st.error(f"Unable to process this image: {e}")

st.divider()

st.caption(
    "Pneumonia Screening System | SVM-based research prototype"
)

