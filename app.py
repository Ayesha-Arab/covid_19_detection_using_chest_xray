# app.py
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import os
import shutil
import time
import requests
import zipfile


st.write(f"TensorFlow Version: {tf.__version__}")

# Define model paths
ORIGINAL_MODEL_PATH = "model_adv.h5"
UPDATED_MODEL_PATH = "model_updated.h5"
BACKUP_DIR = "model_backups"
DATASET_URL = "https://www.dropbox.com/s/7rjw6oet4za01op/CovidDataset-20200427T133042Z-001.zip?dl=1"
DATASET_ZIP = "covid_19.zip"
DATASET_DIR = "CovidDataset"

# Ensure backup directory exists
os.makedirs(BACKUP_DIR, exist_ok=True)

# Global model variable in session state
if 'model' not in st.session_state:
    if os.path.exists(UPDATED_MODEL_PATH):
        st.session_state.model = tf.keras.models.load_model(UPDATED_MODEL_PATH)
        # Build metrics by evaluating on a dummy batch
        dummy_input = np.zeros((1, 224, 224, 3))
        st.session_state.model.predict(dummy_input)
        st.write(f"Loaded updated model from {UPDATED_MODEL_PATH}")
    else:
        st.session_state.model = tf.keras.models.load_model(ORIGINAL_MODEL_PATH)
        dummy_input = np.zeros((1, 224, 224, 3))
        st.session_state.model.predict(dummy_input)
        st.write(f"Loaded original model from {ORIGINAL_MODEL_PATH}")

# Class labels (Binary Classification)
CLASS_NAMES = ["COVID", "Normal"]

# Paths for training data
TRAIN_PATH = os.path.join(DATASET_DIR, "Train")
COVID_DIR = os.path.join(TRAIN_PATH, "Covid")
NORMAL_DIR = os.path.join(TRAIN_PATH, "Normal")

def download_and_extract_dataset():
    if not os.path.exists(DATASET_DIR):
        st.write("Downloading dataset from URL...")
        response = requests.get(DATASET_URL, stream=True)
        if response.status_code == 200:
            with open(DATASET_ZIP, "wb") as f:
                f.write(response.content)
            st.write("Dataset downloaded successfully.")
            try:
                with zipfile.ZipFile(DATASET_ZIP, "r") as zip_ref:
                    zip_ref.extractall(".")
                st.write("Dataset extracted successfully.")
                os.remove(DATASET_ZIP)
                os.makedirs(os.path.join(TRAIN_PATH, "Covid"), exist_ok=True)
                os.makedirs(os.path.join(TRAIN_PATH, "Normal"), exist_ok=True)
            except zipfile.BadZipFile:
                st.error("Downloaded file is not a valid ZIP file. Please check the URL.")
                os.remove(DATASET_ZIP)
        else:
            st.error(f"Failed to download dataset. Status code: {response.status_code}")
    else:
        st.write("Dataset already exists locally.")

download_and_extract_dataset()

st.title("🦠 COVID-19 Detection System")

def preprocess_image(image_pil):
    img = image_pil.resize((224, 224))
    img = img.convert("RGB")
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def backup_model():
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(BACKUP_DIR, f"model_backup_{timestamp}.h5")
    if os.path.exists(UPDATED_MODEL_PATH):
        shutil.copy(UPDATED_MODEL_PATH, backup_path)
    else:
        shutil.copy(ORIGINAL_MODEL_PATH, backup_path)
    st.write(f"Backup created at: {backup_path}")

def retrain_model(last_uploaded_image=None):
    try:
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
        )
        train_generator = train_datagen.flow_from_directory(
            TRAIN_PATH,
            target_size=(224, 224),
            batch_size=16,  # Smaller batch size to emphasize new data
            class_mode='binary',
            shuffle=True
        )
        # Recompile with a moderate learning rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # Adjusted for better adaptation
        st.session_state.model.compile(
            loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
        )
        # Retrain with more emphasis on new data
        st.session_state.model.fit(
            train_generator,
            steps_per_epoch=max(1, train_generator.samples // 16),
            epochs=15,  # More epochs for learning
            verbose=1
        )
        st.session_state.model.save(UPDATED_MODEL_PATH)
        # Reload the model
        st.session_state.model = tf.keras.models.load_model(UPDATED_MODEL_PATH)
        # Build metrics
        dummy_input = np.zeros((1, 224, 224, 3))
        st.session_state.model.predict(dummy_input)

        # Verify learning on the last uploaded image
        if last_uploaded_image is not None:
            img_array = preprocess_image(last_uploaded_image)
            raw_output = st.session_state.model.predict(img_array)[0][0]
            predicted_class = "Normal" if raw_output >= 0.5 else "COVID"
            st.write(f"Post-retraining prediction on last image: {predicted_class}")
        
        st.success(f"Model retrained and saved as {UPDATED_MODEL_PATH}!")
    except Exception as e:
        st.error(f"Error during retraining: {str(e)}")

if uploaded_file := st.file_uploader("Upload a Chest X-ray Image", type=["jpg", "png", "jpeg"]):
    image_pil = Image.open(uploaded_file)
    st.image(image_pil, caption="Uploaded Image", use_container_width=True)
    img_array = preprocess_image(image_pil)
    raw_output = st.session_state.model.predict(img_array)[0][0]
    threshold = 0.5
    if raw_output >= threshold:
        predicted_class = "Normal"
        confidence = raw_output * 100
    else:
        predicted_class = "COVID"
        confidence = (1 - raw_output) * 100

    st.markdown(f"### 🏥 Prediction: **{predicted_class}**")
    st.markdown(f"### 🔍 Confidence: **{confidence:.2f}%**")

    with st.expander("🔍 Debugging Info"):
        st.write(f"🔹 **Image Array Shape:** {img_array.shape}")
        st.write(f"🔹 **Raw Model Output:** {raw_output:.6f}")
        st.write(f"🔹 **Class Mapping:** {CLASS_NAMES}")

    st.markdown("### Feedback")
    feedback = st.radio("Is the prediction correct?", ("Yes", "No"))

    if feedback == "No":
        correct_label = st.selectbox("What is the correct label?", ["COVID", "Normal"])
        if correct_label == "COVID":
            save_path = os.path.join(COVID_DIR, f"{uploaded_file.name}")
        else:
            save_path = os.path.join(NORMAL_DIR, f"{uploaded_file.name}")
        image_pil.save(save_path)
        st.write(f"Image saved to: {save_path}")

        if st.button("Retrain Model"):
            backup_model()
            retrain_model(image_pil)  # Pass the image for verification
    elif feedback == "Yes":
        st.success("Thank you for the feedback! No retraining needed.")