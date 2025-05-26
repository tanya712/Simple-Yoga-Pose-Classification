import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import json
import sys

# Load trained model
model = tf.keras.models.load_model("models/yoga_pose_classifier.h5")

# Load class labels
with open("models/class_labels.json", "r") as f:
    class_labels = json.load(f)  # {0: ["Vrikshasana", "Tree Pose"], 1: ["Virabhadrasana", "Warrior Pose"]}

def predict_pose(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    
    sanskrit_name, english_name = class_labels[str(class_index)]
    
    return sanskrit_name, english_name

# Test prediction
if __name__ == "__main__":
    img_path = sys.argv[1]  # Pass image path as argument
    sanskrit, english = predict_pose(img_path)
    print(f"Predicted Yoga Pose: {english} ({sanskrit})")
