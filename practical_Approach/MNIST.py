import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# Load and preprocess data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Reshape and normalize for CNN
X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0

# Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# Compile and train
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, 
                    epochs=5, 
                    validation_split=0.1,
                    verbose=1)

# Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.3f} (>95% target: {'✅ Achieved' if test_acc > 0.95 else '❌ Not achieved'})")

# Visualize predictions
predictions = model.predict(X_test)
fig, axes = plt.subplots(1, 5, figsize=(12, 3))

for i in range(5):
    sample_idx = np.random.randint(0, X_test.shape[0])
    axes[i].imshow(X_test[sample_idx].reshape(28, 28), cmap='gray')
    pred_label = np.argmax(predictions[sample_idx])
    true_label = y_test[sample_idx]
    
    color = 'green' if pred_label == true_label else 'red'
    axes[i].set_title(f'True: {true_label}\nPred: {pred_label}', color=color)
    axes[i].axis('off')

plt.tight_layout()
plt.show()