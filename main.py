
from pathlib import Path
import json

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps


# ==================================================
# 1. CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR / "models" / "plant_disease_multiplant_model.keras"
)

CLASS_NAMES_PATH = (
    BASE_DIR / "models" / "class_names.json"
)

IMAGE_SIZE = (160, 160)


# ==================================================
# 2. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Plant Doctor AI",
    page_icon="🌿",
    layout="wide",
)

st.title("🌿 Plant Doctor AI")
st.markdown(
    "### AI-powered plant disease detection"
)
st.write(
    "Upload a clear leaf image to identify supported "
    "plant diseases and view basic care recommendations."
)


# ==================================================
# 3. LOAD MODEL
# ==================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )


# ==================================================
# 4. LOAD CLASS LABELS
# ==================================================

@st.cache_data
def load_class_names():
    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"Class labels not found: {CLASS_NAMES_PATH}"
        )

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        names = json.load(file)

    if isinstance(names, dict):
        try:
            names = [
                names[str(i)] if str(i) in names else names[i]
                for i in range(len(names))
            ]
        except (KeyError, TypeError):
            names = list(names.values())

    if not isinstance(names, list):
        raise ValueError(
            "class_names.json must contain a list or dictionary."
        )

    return names


# ==================================================
# 5. PREPROCESS IMAGE
# ==================================================

def preprocess_image(image):
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.asarray(
        image,
        dtype=np.float32,
    )

    # The model already contains a Rescaling layer.
    # Do not normalize the image a second time.

    image_array = np.expand_dims(
        image_array,
        axis=0,
    )

    return image_array


# ==================================================
# 6. PREDICT DISEASE
# ==================================================

def predict_disease(model, class_names, image):
    processed_image = preprocess_image(image)

    predictions = model.predict(
        processed_image,
        verbose=0,
    )[0]

    if len(predictions) != len(class_names):
        raise ValueError(
            f"Model has {len(predictions)} outputs, "
            f"but {len(class_names)} class labels were loaded."
        )

    if not np.all(np.isfinite(predictions)):
        raise ValueError(
            "Model returned invalid prediction values."
        )

    # The model is expected to return softmax probabilities.
    if (
        np.any(predictions < 0)
        or np.any(predictions > 1)
        or not np.isclose(np.sum(predictions), 1.0, atol=0.01)
    ):
        raise ValueError(
            "Model output is not a valid probability distribution. "
            "Please verify the model architecture."
        )

    top_indices = np.argsort(predictions)[::-1][:3]

    top_predictions = []

    for index in top_indices:
        top_predictions.append({
            "class_name": class_names[int(index)],
            "confidence": float(predictions[index]),
        })

    predicted_index = int(np.argmax(predictions))
    predicted_class = class_names[predicted_index]
    confidence = float(predictions[predicted_index])

    return predicted_class, confidence, top_predictions


# ==================================================
# 7. FORMAT CLASS LABEL
# ==================================================

def format_label(class_name):
    name = class_name

    replacements = {
        "Pepper__bell___Bacterial_spot": "Bacterial Spot",
        "Pepper__bell___healthy": "Healthy",
        "Potato___Early_blight": "Early Blight",
        "Potato___Late_blight": "Late Blight",
        "Potato___healthy": "Healthy",
        "Tomato__Target_Spot": "Target Spot",
        "Tomato__Tomato_mosaic_virus": "Tomato Mosaic Virus",
        "Tomato__Tomato_YellowLeaf__Curl_Virus":
            "Tomato Yellow Leaf Curl Virus",
        "Tomato_Bacterial_spot": "Bacterial Spot",
        "Tomato_Early_blight": "Early Blight",
        "Tomato_healthy": "Healthy",
        "Tomato_Late_blight": "Late Blight",
        "Tomato_Leaf_Mold": "Leaf Mold",
        "Tomato_Septoria_leaf_spot": "Septoria Leaf Spot",
        "Tomato_Spider_mites_Two_spotted_spider_mite":
            "Two-Spotted Spider Mites",
    }

    return replacements.get(
        name,
        name.replace("_", " ").replace("__", " ").strip(),
    )


def get_plant_name(class_name):
    if class_name.startswith("Tomato"):
        return "🍅 Tomato"

    if class_name.startswith("Potato"):
        return "🥔 Potato"

    if class_name.startswith("Pepper"):
        return "🫑 Bell Pepper"

    return "🌿 Unknown Plant"


# ==================================================
# 8. CARE RECOMMENDATIONS
# ==================================================

def get_care_recommendations(class_name):
    if "healthy" in class_name.lower():
        return [
            "Continue regular watering according to the plant's needs.",
            "Provide suitable sunlight and good air circulation.",
            "Inspect leaves regularly for new spots or discoloration.",
        ]

    if "Early_blight" in class_name:
        return [
            "Remove badly affected leaves using clean tools.",
            "Avoid overhead watering and keep foliage dry.",
            "Check whether the symptoms spread to nearby leaves.",
            "Confirm the diagnosis before choosing a treatment.",
        ]

    if "Late_blight" in class_name:
        return [
            "Separate affected plants from healthy plants where practical.",
            "Avoid wetting leaves during watering.",
            "Remove severely affected plant material safely.",
            "Seek local agricultural guidance promptly if symptoms spread.",
        ]

    if "Bacterial_spot" in class_name:
        return [
            "Avoid handling plants while foliage is wet.",
            "Clean gardening tools after use.",
            "Avoid overhead irrigation where possible.",
            "Get confirmation before applying any pesticide.",
        ]

    if "mosaic_virus" in class_name:
        return [
            "Inspect the plant for mosaic patterns and leaf distortion.",
            "Clean tools after handling potentially affected plants.",
            "Check for insects that may spread plant viruses.",
            "Consult an agricultural expert about safe plant management.",
        ]

    if "YellowLeaf__Curl_Virus" in class_name:
        return [
            "Inspect the plant for leaf curling and yellowing.",
            "Check the undersides of leaves for whiteflies.",
            "Monitor nearby plants for similar symptoms.",
            "Seek local agricultural advice before treatment.",
        ]

    if "Leaf_Mold" in class_name:
        return [
            "Improve airflow around the plant.",
            "Avoid prolonged moisture on leaf surfaces.",
            "Remove severely affected leaves using clean tools.",
            "Monitor humidity and plant spacing.",
        ]

    if "Septoria_leaf_spot" in class_name:
        return [
            "Remove severely affected lower leaves.",
            "Avoid splashing soil onto leaves during watering.",
            "Keep the area around plants free of infected debris.",
            "Confirm the diagnosis before using a treatment.",
        ]

    if "Spider_mites" in class_name:
        return [
            "Inspect leaf undersides for mites and fine webbing.",
            "Check nearby plants for similar symptoms.",
            "Use an appropriate control method only after confirming the pest.",
            "Follow the product label and local agricultural guidance.",
        ]

    if "Target_Spot" in class_name:
        return [
            "Remove severely affected leaves where appropriate.",
            "Improve airflow and avoid overhead watering.",
            "Monitor new growth for spreading spots.",
            "Confirm the diagnosis before selecting a treatment.",
        ]

    return [
        "Inspect the plant and nearby leaves carefully.",
        "Provide appropriate water, light, and air circulation.",
        "Consult an agricultural expert if symptoms worsen.",
    ]


# ==================================================
# 9. LOAD MODEL AND LABELS
# ==================================================

try:
    model = load_model()
    class_names = load_class_names()

    if model.output_shape[-1] != len(class_names):
        st.error(
            "The number of model outputs does not match "
            "the number of class labels."
        )
        st.stop()

except Exception as error:
    st.error(f"Unable to load the model: {error}")
    st.stop()


# ==================================================
# 10. IMAGE UPLOAD
# ==================================================

st.subheader("📷 Upload a Leaf Image")

uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"],
    help="Use a clear image with the leaf visible.",
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image).convert("RGB")

        left, right = st.columns([1, 1])

        with left:
            st.image(
                image,
                caption="Uploaded leaf image",
                use_container_width=True,
            )

        with right:
            st.subheader("Plant Identification")

            st.subheader("🌱 Supported Plants")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("### 🍅")
                st.markdown("**Tomato**")

            with col2:
                st.markdown("### 🥔")
                st.markdown("**Potato**")

            with col3:
                st.markdown("### 🫑")
                st.markdown("**Bell Pepper**")

            st.caption(
                "Disease detection is currently supported for these three plants."
            )

            if st.button(
                "🔍 Analyze Leaf",
                type="primary",
                use_container_width=True,
                key="analyze_leaf_button",
            ):
                with st.spinner("Analyzing the leaf image..."):
                    predicted_class, confidence, top_predictions = (
                        predict_disease(
                            model,
                            class_names,
                            image,
                        )
                    )

                    st.session_state["plant_result"] = {
                        "predicted_class": predicted_class,
                        "confidence": confidence,
                        "top_predictions": top_predictions,
                    }

        # Display the most recent prediction.

        # Display the most recent prediction.
        if "plant_result" in st.session_state:
            result = st.session_state["plant_result"]

            predicted_class = result["predicted_class"]
            confidence = result["confidence"]
            top_predictions = result["top_predictions"]

            st.divider()
            st.subheader("🌱 Diagnosis Results")

            result_left, result_right = st.columns(2)

            with result_left:
                st.markdown("**Predicted Plant**")
                st.title(get_plant_name(predicted_class))

                st.markdown("**Predicted Category**")
                st.subheader(format_label(predicted_class))

            with result_right:
                st.markdown("**Model Confidence**")
                st.metric(
                    label="Top prediction",
                    value=f"{confidence * 100:.2f}%",
                )

                if confidence < 0.60:
                    st.warning(
                        "The prediction confidence is low. "
                        "Try a clearer image and seek confirmation."
                    )
                elif confidence < 0.80:
                    st.info(
                        "Consider checking the top predictions "
                        "before drawing a conclusion."
                    )

            st.subheader("📊 Top 3 Predictions")

            for item in top_predictions:
                label = format_label(item["class_name"])
                plant = get_plant_name(item["class_name"])
                score = item["confidence"]

                st.write(
                    f"**{plant} — {label}:** {score * 100:.2f}%"
                )
                st.progress(
                    min(max(score, 0.0), 1.0)
                )

            st.subheader("🪴 Care Recommendations")

            for recommendation in get_care_recommendations(
                predicted_class
            ):
                st.markdown(f"- {recommendation}")

            st.caption(
                "Important: This AI result is an estimate, not a "
                "confirmed agricultural diagnosis. Image quality, "
                "lighting, and real-world conditions can affect results."
            )

    except Exception as error:
        st.error(f"Unable to analyze the image: {error}")

else:
    st.info(
        "Upload a leaf image to get started."
    )

st.divider()
st.caption(
    "Plant Doctor AI | Educational plant disease detection project"
)
