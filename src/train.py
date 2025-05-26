import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from preprocess import train_generator, val_generator

# Load MobileNetV2 as base model
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze layers

# Add custom layers
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dense(len(train_generator.class_indices), activation='softmax')(x)

# Create model
model = Model(inputs=base_model.input, outputs=x)

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
epochs = 10
model.fit(train_generator, validation_data=val_generator, epochs=epochs)

# Save model
model.save("models/yoga_pose_classifier.h5")
print("Model saved successfully!")
