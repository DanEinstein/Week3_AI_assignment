import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Load and preprocess data - FIXED
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# FIX 1: Proper reshape for CNN
X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0

# FIX 2: Correct CNN architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),  # Correct shape
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),  # Correct position: after conv layers
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# FIX 3: Correct loss function
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # For integer labels
              metrics=['accuracy'])

# Now runs without errors
model.fit(X_train, y_train, epochs=3, validation_split=0.1)


#  Bugs Fixed:
# Dimension Mismatch: Added .reshape(-1, 28, 28, 1) for proper CNN input

# Input Shape: Changed from (784,) to (28, 28, 1) for Conv2D layers

# Layer Order: Moved Flatten() to after convolutional layers

# Loss Function: Changed to sparse_categorical_crossentropy for integer labels

# Architecture: Added missing Conv2D and Dropout layers for better performance