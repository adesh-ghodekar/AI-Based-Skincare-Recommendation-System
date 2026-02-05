#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Constants
CLASS_NAMES = ['Oily', 'Normal to Dry', 'Acne-Prone', 'Fine lines_wrinkles']
PRIMARY_CLASSES = ['Oily', 'Normal to Dry', 'Acne-Prone']
SECONDARY_CONDITIONS = ['Fine lines_wrinkles']
IMG_SIZE = (224, 224)
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'skin_type_model.h5')  # HDF5 format

# Load model (ensure it's already trained)
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Trained model not found at {MODEL_PATH}. Please train the model first using train_skin_model.py.")

model = load_model(MODEL_PATH)

def preprocess_image(img_path):
    """Preprocess image for prediction"""
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

def detect_skin_type(image_path, model_instance=None, combination_threshold=0.15):
    """Detect skin type and secondary conditions"""
    global model
    if model_instance is None:
        model_instance = model
    if model_instance is None:
        raise ValueError("Model not loaded.")

    img_array = preprocess_image(image_path)
    predictions = model_instance.predict(img_array)[0]

    # Primary skin type logic
    primary_scores = {label: predictions[CLASS_NAMES.index(label)] for label in PRIMARY_CLASSES}
    primary_type = max(primary_scores, key=primary_scores.get)

    # Combination skin check
    oily = predictions[CLASS_NAMES.index('Oily')]
    dry = predictions[CLASS_NAMES.index('Normal to Dry')]
    if abs(oily - dry) < combination_threshold and primary_type in ['Oily', 'Normal to Dry']:
        primary_type = 'Combination'

    # Detect additional conditions like wrinkles
    conditions = []
    for label in SECONDARY_CONDITIONS:
        idx = CLASS_NAMES.index(label)
        prob = predictions[idx]
        if prob >= 0.28:
            conditions.append(label)

    return primary_type, conditions, predictions.tolist()

# Example usage
if __name__ == "__main__":
    image_path = "Photo on 21-03-25 at 10.21 PM.jpg"  # Replace with your image path
    skin_type, conditions, probabilities = detect_skin_type(image_path)

    print("\n🧴 Skin Analysis Results:")
    print(f"🔹 Primary Skin Type: {skin_type}")
    print(f"🔸 Additional Conditions: {', '.join(conditions) if conditions else 'None'}")

    print("\n📊 Detailed Probabilities:")
    for name, prob in zip(CLASS_NAMES, probabilities):
        print(f"{name}: {prob*100:.1f}%")
