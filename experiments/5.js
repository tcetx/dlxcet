export default {
  name: "EXP 5: Autoencoder Model for Image Compression",
  
  theory: `An autoencoder is a neural network designed to learn efficient representations of data for dimensionality reduction or compression. 
It consists of two main components: 
1. Encoder: Compresses input data into a smaller latent representation (bottleneck). 
2. Decoder: Reconstructs the original input from the compressed latent vector. 

In this experiment, the MNIST dataset (28x28 grayscale images) is used. 
The encoder compresses the image to a 64-dimensional latent vector, while the decoder reconstructs the image from this compressed form. 
The network is trained to minimize reconstruction loss, allowing us to compare original and reconstructed images to visualize how much information is preserved during compression.`,

  code: `
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.datasets import mnist

# Load MNIST dataset
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test  = x_test.astype('float32') / 255.

# Flatten images
x_train_flat = x_train.reshape((len(x_train), 28*28))
x_test_flat  = x_test.reshape((len(x_test), 28*28))

# Define Autoencoder
input_dim = 28*28
encoding_dim = 64

# Encoder
input_img = Input(shape=(input_dim,))
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(encoding_dim, activation='relu')(encoded)

# Decoder
decoded = Dense(128, activation='relu')(encoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

# Autoencoder Model
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train the model
history = autoencoder.fit(x_train_flat, x_train_flat,
                          epochs=50,
                          batch_size=256,
                          shuffle=True,
                          validation_data=(x_test_flat, x_test_flat))

# Evaluate reconstruction
decoded_imgs = autoencoder.predict(x_test_flat)

# Plot original and reconstructed images
n = 10
plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test_flat[i].reshape(28,28), cmap='gray')
    plt.axis('off')
    ax = plt.subplot(2, n, i+n+1)
    plt.imshow(decoded_imgs[i].reshape(28,28), cmap='gray')
    plt.axis('off')
plt.show()

# Plot training loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Autoencoder Training Loss")
plt.legend()
plt.show()
  `,

  algorithm: `
1. Load and normalize the MNIST dataset
2. Flatten images for fully connected autoencoder
3. Define encoder network to compress input to 64-dimensional latent vector
4. Define decoder network to reconstruct input from latent vector
5. Compile autoencoder with binary cross-entropy loss
6. Train the model on training data and validate on test data
7. Predict reconstructed images on test set
8. Visualize original and reconstructed images
9. Plot training loss to evaluate learning
  `,

  viva: [
    {
      q: "What is the purpose of the bottleneck layer in an autoencoder?",
      a: "The bottleneck layer, also called the latent space or encoding layer, is the compressed representation of the input data. Its purpose is to reduce the dimensionality of the input, forcing the network to learn the most important features and patterns. By compressing information into a smaller space, the autoencoder learns efficient data representations that capture essential structures while discarding redundant information. This allows for tasks like compression, denoising, and feature extraction."
    },
    {
      q: "Why do we flatten images before passing them to a fully connected autoencoder?",
      a: "Fully connected layers expect 1D input vectors. Flattening converts the 2D image (28x28) into a 1D array (784 values), enabling the dense layers to process each pixel as a separate input. This transformation is necessary for fully connected autoencoders, which do not inherently handle spatial relationships. Flattening allows the network to learn representations across the entire image but loses explicit spatial structure, which is acceptable for simple compression tasks."
    },
    {
      q: "How do we evaluate the performance of an autoencoder?",
      a: "Autoencoder performance is evaluated by comparing the reconstructed output to the original input, typically using reconstruction loss such as Mean Squared Error or Binary Cross-Entropy. Lower reconstruction loss indicates the network successfully preserves important information. Visualization of reconstructed images alongside originals helps qualitatively assess how much detail is retained. Additionally, the latent representation can be analyzed for features or used in downstream tasks like classification or anomaly detection."
    }
  ]
};
