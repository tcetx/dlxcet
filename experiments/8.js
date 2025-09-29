export default {
  name: "EXP 8: Digit Recognition using CNN",

  theory: `Convolutional Neural Networks (CNNs) are designed for image data and can automatically learn spatial hierarchies of features. 
They extract local patterns like edges, curves, and shapes via convolutional layers. Pooling layers reduce dimensionality while preserving key information.
The MNIST dataset provides 28×28 grayscale images of handwritten digits (0–9). 
CNN layers extract features, followed by dense layers for classification. The final softmax layer outputs probabilities for each digit class.`,

  code: `
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize and reshape
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32') / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test  = x_test.reshape(-1, 28, 28, 1)

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test  = to_categorical(y_test, 10)

# Build CNN model
model = Sequential([
    Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, kernel_size=(3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(x_train, y_train, epochs=5, batch_size=128,
                    validation_data=(x_test, y_test))

# Evaluate on test data
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy: {acc*100:.2f}%")

# Plot accuracy and loss
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label="Train Acc")
plt.plot(history.history['val_accuracy'], label="Val Acc")
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Val Loss")
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Test prediction on a sample
sample = x_test[0].reshape(1,28,28,1)
pred = model.predict(sample)
print("Predicted Digit:", np.argmax(pred))

# Show actual image
plt.imshow(x_test[0].reshape(28,28), cmap="gray")
plt.title(f"Actual Digit: {np.argmax(y_test[0])}")
plt.axis("off")
plt.show()
  `,

  algorithm: `
1. Load MNIST dataset
2. Normalize pixel values to range [0,1] and reshape for CNN input
3. One-hot encode labels
4. Build CNN with Conv2D, MaxPooling2D, Flatten, Dense, and Dropout layers
5. Compile model with Adam optimizer and categorical cross-entropy loss
6. Train model with validation on test set
7. Evaluate accuracy on test set
8. Plot training/validation accuracy and loss
9. Predict digit for a sample image and display it
  `,

  viva: [
    {
      q: "Why use CNNs instead of fully connected networks for images?",
      a: "CNNs automatically extract spatial features like edges and patterns, reducing the number of parameters and improving performance on image tasks."
    },
    {
      q: "What is the purpose of pooling layers?",
      a: "Pooling layers reduce spatial dimensions, keep important features, and make the model less sensitive to translations or distortions."
    },
    {
      q: "Why do we normalize the image data?",
      a: "Normalization scales pixel values to [0,1], helping faster and more stable training by preventing large activations."
    },
    {
      q: "What does the softmax output layer do?",
      a: "It converts raw output scores into probabilities across the 10 digit classes, allowing selection of the most likely digit."
    }
  ]
};
