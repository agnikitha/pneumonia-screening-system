
import gradio as gr
import numpy as np
import pandas as pd
import joblib
import json
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import sobel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load trained model and preprocessing objects
model = joblib.load(os.path.join(BASE_DIR, "svm_pneumonia_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "feature_scaler.pkl"))

with open(os.path.join(BASE_DIR, "selected_features.json"), "r") as f:
    selected_features = json.load(f)

def extract_prediction_features(image):
    # Convert to grayscale
    image = image.convert("L")

    # Resize to PneumoniaMNIST size
    image = image.resize((28, 28))

    # Normalize exactly as used during training
    img = np.asarray(image, dtype=np.float32) / 255.0

    features = {}

    features["mean_intensity"] = np.mean(img)
    features["std_intensity"] = np.std(img)
    features["min_intensity"] = np.min(img)
    features["max_intensity"] = np.max(img)
    features["median_intensity"] = np.median(img)

    features["percentile_10"] = np.percentile(img, 10)
    features["percentile_25"] = np.percentile(img, 25)
    features["percentile_75"] = np.percentile(img, 75)
    features["percentile_90"] = np.percentile(img, 90)

    # Entropy
    hist, _ = np.histogram(img, bins=256, range=(0, 1))
    prob = hist / np.sum(hist)
    prob = prob[prob > 0]
    features["entropy"] = -np.sum(prob * np.log2(prob))

    # GLCM
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

    # Edge density
    edges = sobel(img)
    threshold = np.mean(edges)
    features["edge_density"] = np.mean(edges > threshold)

    return features


def predict_pneumonia(image):

    if image is None:
        return "No image uploaded", "", ""

    features = extract_prediction_features(image)

    # Exact feature order selected during training
    X = pd.DataFrame(
        [[features[f] for f in selected_features]],
        columns=selected_features
    )

    # Scale features
    X_scaled = scaler.transform(X.values)

    # Prediction
    prediction = model.predict(X_scaled)[0]

    # Probabilities
    probabilities = model.predict_proba(X_scaled)[0]

    normal_probability = probabilities[0] * 100
    pneumonia_probability = probabilities[1] * 100

    if prediction == 1:
        result = "PNEUMONIA"
        recommendation = (
            "The model classified this image as pneumonia-positive. "
            "Further clinical and radiological evaluation is recommended."
        )
    else:
        result = "NORMAL"
        recommendation = (
            "The model classified this image as normal. "
            "Clinical assessment should still be considered when symptoms are present."
        )

    probability_text = (
        f"Normal probability: {normal_probability:.2f}%\n"
        f"Pneumonia probability: {pneumonia_probability:.2f}%"
    )

    return result, probability_text, recommendation


# Gradio interface
with gr.Blocks(title="Pneumonia Screening System") as app:

    gr.Markdown(
        "# Pneumonia Screening System\n"
        "Computer-aided pneumonia screening from chest X-ray images "
        "using image features and an SVM classifier."
    )

    with gr.Row():

        with gr.Column():
            image_input = gr.Image(
                type="pil",
                label="Upload Chest X-ray"
            )

            submit_button = gr.Button(
                "Submit",
                variant="primary"
            )

            clear_button = gr.Button("Clear")

        with gr.Column():
            result_output = gr.Textbox(
                label="Screening Result"
            )

            probability_output = gr.Textbox(
                label="Prediction Probabilities"
            )

            recommendation_output = gr.Textbox(
                label="Clinical Support Recommendation"
            )

    submit_button.click(
        fn=predict_pneumonia,
        inputs=image_input,
        outputs=[
            result_output,
            probability_output,
            recommendation_output
        ]
    )

    clear_button.click(
        fn=lambda: (None, "", "", ""),
        inputs=None,
        outputs=[
            image_input,
            result_output,
            probability_output,
            recommendation_output
        ]
    )


if __name__ == "__main__":
    app.launch()
