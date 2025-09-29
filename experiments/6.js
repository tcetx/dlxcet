export default {
  name: "EXP 6: Image Denoising using Autoencoder",
  
  theory: `Image denoising autoencoders are neural networks designed to remove noise from images. 
The model learns to map noisy input images to their clean versions. 
It consists of two main parts:
1. Encoder: Compresses noisy input images into a latent representation.
2. Decoder: Reconstructs the clean image from the latent vector.

The network is trained to minimize reconstruction loss between the output (denoised image) and the original clean image. 
This helps the autoencoder learn noise-invariant features and preserve important image structures.`,

  code: `
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
from tensorflow.keras.datasets import mnist

# Load MNIST dataset
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test  = x_test.astype('float32') / 255.

# Add random noise
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy  = x_test  + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy  = np.clip(x_test_noisy, 0., 1.)

# Flatten images
x_train_flat = x_train.reshape((len(x_train), 28*28))
x_test_flat  = x_test.reshape((len(x_test), 28*28))
x_train_noisy_flat = x_train_noisy.reshape((len(x_train_noisy), 28*28))
x_test_noisy_flat  = x_test_noisy.reshape((len(x_test_noisy), 28*28))

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
history = autoencoder.fit(x_train_noisy_flat, x_train_flat,
                          epochs=50,
                          batch_size=256,
                          shuffle=True,
                          validation_data=(x_test_noisy_flat, x_test_flat))

# Evaluate reconstruction
decoded_imgs = autoencoder.predict(x_test_noisy_flat)

# Plot original, noisy, and reconstructed images
n = 10
plt.figure(figsize=(20,6))
for i in range(n):
    # Original
    ax = plt.subplot(3, n, i+1)
    plt.imshow(x_test_flat[i].reshape(28,28), cmap='gray')
    plt.axis('off')
    # Noisy
    ax = plt.subplot(3, n, i+n+1)
    plt.imshow(x_test_noisy_flat[i].reshape(28,28), cmap='gray')
    plt.axis('off')
    # Reconstructed
    ax = plt.subplot(3, n, i+2*n+1)
    plt.imshow(decoded_imgs[i].reshape(28,28), cmap='gray')
    plt.axis('off')
plt.show()

# Plot training loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Denoising Autoencoder Training Loss")
plt.legend()
plt.show()
  `,

  algorithm: `
1. Load and normalize MNIST dataset
2. Add Gaussian noise to input images
3. Flatten images for fully connected autoencoder
4. Define encoder network to compress noisy input to latent space
5. Define decoder network to reconstruct clean images
6. Compile autoencoder with binary cross-entropy loss
7. Train model with noisy inputs and original images as targets
8. Predict denoised images on test set
9. Visualize original, noisy, and reconstructed images
10. Plot training loss for evaluation
  `,

  viva: [
    {
      q: "Why do we add noise to the input images in a denoising autoencoder?",
      a: "Adding noise forces the autoencoder to learn robust features that capture the essential structure of the data, rather than memorizing the input. The network learns to remove noise and reconstruct the original clean image, improving generalization and making it useful for real-world noisy data."
    },
    {
      q: "How is a denoising autoencoder different from a standard autoencoder?",
      a: "A standard autoencoder reconstructs its input and compresses it to a latent representation, while a denoising autoencoder takes a corrupted (noisy) input and tries to reconstruct the clean original. This teaches the model to be noise-invariant and improves its robustness."
    },
    {
      q: "What loss function is used for image reconstruction and why?",
      a: "Binary Cross-Entropy (BCE) or Mean Squared Error (MSE) is commonly used. BCE is suitable for normalized pixel values (0 to 1), treating each pixel as a probability. MSE measures the average squared difference between original and reconstructed pixels. Both losses quantify reconstruction quality."
    }
  ]
};
