
import os
import json
import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "model/plant_disease_model.keras"
CLASS_NAMES_PATH = "model/class_names.json"
DISEASE_INFO_PATH = "disease_info.json"

IMAGE_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.60


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌱",
    layout="centered"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        return None

    return tf.keras.models.load_model(MODEL_PATH)


# ============================================================
# LOAD CLASS NAMES
# ============================================================

@st.cache_data
def load_class_names():

    if not os.path.exists(CLASS_NAMES_PATH):
        return []

    with open(
        CLASS_NAMES_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# LOAD DISEASE INFORMATION
# ============================================================

@st.cache_data
def load_disease_info():

    if not os.path.exists(DISEASE_INFO_PATH):
        return {}

    with open(
        DISEASE_INFO_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# PREPARE IMAGE
# ============================================================

def prepare_image(image):

    image = image.convert("RGB")

    image = image.resize(IMAGE_SIZE)

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    image_array = preprocess_input(
        image_array
    )

    return image_array


# ============================================================
# MAKE PREDICTION
# ============================================================

def predict_disease(
    model,
    image,
    class_names
):

    processed_image = prepare_image(
        image
    )

    predictions = model.predict(
        processed_image,
        verbose=0
    )[0]

    predicted_index = int(
        np.argmax(predictions)
    )

    predicted_class = class_names[
        predicted_index
    ]

    confidence = float(
        predictions[predicted_index]
    )

    return (
        predicted_class,
        confidence,
        predictions
    )


# ============================================================
# DISPLAY BULLET LIST
# ============================================================

def show_list(title, items):

    st.subheader(title)

    for item in items:
        st.markdown(
            "• " + item
        )


# ============================================================
# TITLE
# ============================================================

st.title(
    "🌱 Plant Disease Detection System"
)

st.write(
    "Upload a photograph of a plant leaf "
    "and the AI model will predict the "
    "most likely disease category."
)

st.info(
    "This is an educational AI project. "
    "The prediction is not a professional "
    "agricultural diagnosis."
)


# ============================================================
# LOAD FILES
# ============================================================

model = load_model()

class_names = load_class_names()

disease_info = load_disease_info()


# ============================================================
# CHECK MODEL
# ============================================================

if model is None:

    st.error(
        "❌ AI model not found."
    )

    st.write(
        "First run:"
    )

    st.code(
        "python train_model.py",
        language="bash"
    )

    st.stop()


# ============================================================
# CHECK CLASS NAMES
# ============================================================

if len(class_names) == 0:

    st.error(
        "❌ class_names.json could not be loaded."
    )

    st.write(
        "Check that this file exists:"
    )

    st.code(
        "model/class_names.json"
    )

    st.stop()


# ============================================================
# CHECK DISEASE INFORMATION
# ============================================================

if len(disease_info) == 0:

    st.error(
        "❌ disease_info.json could not be loaded."
    )

    st.write(
        "Check that disease_info.json is in "
        "the same folder as app.py."
    )

    st.stop()


# ============================================================
# SHOW TRAINED CLASSES
# ============================================================

with st.expander(
    "📚 AI trained categories"
):

    for i, name in enumerate(
        class_names
    ):

        st.write(
            f"{i + 1}. {name.replace('_', ' ')}"
        )


# ============================================================
# UPLOAD IMAGE
# ============================================================

st.subheader(
    "📷 Upload Plant Leaf"
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp",
        "bmp"
    ]
)


# ============================================================
# IMAGE PROCESSING
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    )

    st.image(
        image,
        caption="Uploaded plant leaf",
        use_container_width=True
    )

    st.success(
        "Image uploaded successfully!"
    )

    # --------------------------------------------------------
    # DETECTION BUTTON
    # --------------------------------------------------------

    if st.button(
        "🔍 Detect Plant Disease",
        type="primary",
        use_container_width=True
    ):

        st.write(
            "🤖 Analysing image..."
        )

        try:

            (
                predicted_class,
                confidence,
                predictions
            ) = predict_disease(
                model,
                image,
                class_names
            )

        except Exception as error:

            st.error(
                "❌ An error occurred while "
                "analysing the image."
            )

            st.code(
                str(error)
            )

            st.stop()


        # ====================================================
        # RESULT
        # ====================================================

        st.divider()

        st.header(
            "🔬 Detection Result"
        )

        readable_name = (
            predicted_class
            .replace("_", " ")
        )


        # ====================================================
        # PREDICTION
        # ====================================================

        st.subheader(
            "Predicted Category"
        )

        st.success(
            readable_name
        )


        # ====================================================
        # CONFIDENCE
        # ====================================================

        st.subheader(
            "AI Confidence"
        )

        confidence_percent = (
            confidence * 100
        )

        st.progress(
            min(
                max(
                    confidence,
                    0.0
                ),
                1.0
            )
        )

        st.write(
            f"**{confidence_percent:.2f}%**"
        )


        # ====================================================
        # LOW CONFIDENCE WARNING
        # ====================================================

        if confidence < CONFIDENCE_THRESHOLD:

            st.warning(
                "⚠️ The AI is not very confident "
                "about this prediction."
            )

            st.write(
                "Try uploading a clearer image "
                "with good lighting and the leaf "
                "taking up most of the photograph."
            )


        # ====================================================
        # DISEASE INFORMATION
        # ====================================================

        info = disease_info.get(
            predicted_class
        )


        if info is not None:

            st.divider()

            # ------------------------------------------------
            # DESCRIPTION
            # ------------------------------------------------

            st.subheader(
                "📖 About the Disease"
            )

            st.write(
                info.get(
                    "description",
                    "No description available."
                )
            )


            # ------------------------------------------------
            # SYMPTOMS
            # ------------------------------------------------

            symptoms = info.get(
                "symptoms",
                []
            )

            if symptoms:

                show_list(
                    "🔎 Common Symptoms",
                    symptoms
                )


            # ------------------------------------------------
            # SOLUTIONS
            # ------------------------------------------------

            solutions = info.get(
                "solutions",
                []
            )

            if solutions:

                show_list(
                    "🛠 Suggested Solutions",
                    solutions
                )


            # ------------------------------------------------
            # PREVENTION
            # ------------------------------------------------

            prevention = info.get(
                "prevention",
                []
            )

            if prevention:

                show_list(
                    "🛡 Prevention Tips",
                    prevention
                )


        else:

            st.warning(
                "No disease information was "
                "found for this category."
            )


        # ====================================================
        # TOP 3 PREDICTIONS
        # ====================================================

        st.divider()

        st.subheader(
            "📊 Other Possible Categories"
        )

        top_indices = np.argsort(
            predictions
        )[::-1]

        top_indices = top_indices[
            :min(
                3,
                len(class_names)
            )
        ]


        for index in top_indices:

            category = (
                class_names[
                    int(index)
                ]
                .replace("_", " ")
            )

            probability = float(
                predictions[
                    int(index)
                ]
            )

            st.write(
                f"**{category}** — "
                f"{probability * 100:.2f}%"
            )


# ============================================================
# NO IMAGE
# ============================================================

else:

    st.info(
        "👆 Upload a plant leaf image "
        "to start the detection."
    )


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.caption(
    "🌱 Plant Disease Detection System | "
    "Class 11 AI Capstone Project"
)

st.caption(
    "This system is intended for educational "
    "demonstration only and should not replace "
    "professional agricultural advice."
)
