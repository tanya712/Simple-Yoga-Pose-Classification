from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import json
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load the trained model
MODEL_PATH = "models/yoga_pose_classifier.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Load class labels
LABELS_PATH = "models/class_labels.json"
with open(LABELS_PATH, "r") as f:
    class_labels = json.load(f)  # {0: ["Vrikshasana", "Tree Pose"], ...}

# Load pose descriptions
DESCRIPTION_PATH = "models/pose_descriptions.json"
with open(DESCRIPTION_PATH, "r") as f:
    pose_descriptions = json.load(f)  # { "Adho Mukha Svanasana": "Description", ... }

# Function to preprocess image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error="No file uploaded.")

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="No file selected.")

    # Save file securely
    filename = secure_filename(file.filename)
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    # Preprocess and predict
    img_array = preprocess_image(file_path)
    prediction = model.predict(img_array)[0]
    class_index = int(np.argmax(prediction))
    confidence = float(prediction[class_index]) * 100

    # Get pose names
    sanskrit_name, english_name = class_labels[str(class_index)]

    # Get description from dictionary
    description = pose_descriptions.get(english_name, "Description not available.")

    return render_template('index.html',
                           prediction=english_name,
                           sanskrit=sanskrit_name,
                           confidence=round(confidence, 2),
                           filename=filename,
                           description=description)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
