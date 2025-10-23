import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Load data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# BUG 1: Missing reshape for CNN input
X_train = X_train / 255.0
X_test = X_test / 255.0

# Build model with BUG 2: Wrong input shape
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(784,)),  # Wrong shape
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),  # BUG 3: Flatten in wrong position
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# BUG 4: Wrong loss function for integer labels
model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # Should be sparse_categorical_crossentropy
              metrics=['accuracy'])

# This will crash with dimension errors
model.fit(X_train, y_train, epochs=3)
