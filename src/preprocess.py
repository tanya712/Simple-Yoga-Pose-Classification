from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
import os

dataset_dir = "dataset"
json_path = os.path.join(dataset_dir, "Poses.json")
img_size = (224, 224)
batch_size = 32

# Load pose name mapping from JSON
with open(json_path, "r") as f:
    poses_data = json.load(f)["Poses"]

# Normalize folder names (remove spaces & convert to lowercase)
pose_names = {pose["sanskrit_name"].replace(" ", "").lower(): (pose["sanskrit_name"], pose["english_name"]) for pose in poses_data}

# Data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  # 80% training, 20% validation
)

# Training data generator
train_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

# Validation data generator 
val_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Get class labels from dataset
class_labels = {v: k for k, v in train_generator.class_indices.items()}  # {0: "Adho Mukha Svanasana", 1: "Vrikshasana"}

# Normalize dataset folder names to match `pose_names`
normalized_class_labels = {idx: pose_names.get(class_labels[idx].replace(" ", "").lower(), ("Unknown", "Unknown")) for idx in class_labels}

# Save updated labels
with open("models/class_labels.json", "w") as f:
    json.dump(normalized_class_labels, f)

print("Updated class labels saved:", normalized_class_labels)
