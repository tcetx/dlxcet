export default {
  name: "EXP 7: Sentiment Analysis using RNN",
  
  theory: `Sentiment analysis aims to determine the emotional tone behind text. 
Word order and context are crucial: for example, "not good" vs "good" have opposite meanings. 
Traditional models like Bag-of-Words ignore sequence information. 
Recurrent Neural Networks (RNNs) process sequential data, maintaining a hidden state to capture past information.
The IMDB dataset contains 25,000 labeled movie reviews (positive/negative). 
RNNs can learn dependencies across words to predict sentiment.`,

  code: `
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
import matplotlib.pyplot as plt

# Parameters
vocab_size = 10000
max_len = 200
embedding_dim = 32

# Load dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

# Pad sequences
x_train = pad_sequences(x_train, maxlen=max_len)
x_test  = pad_sequences(x_test,  maxlen=max_len)

# Build RNN model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
    SimpleRNN(64, activation='tanh'),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(x_train, y_train,
                    epochs=5,
                    batch_size=64,
                    validation_split=0.2)

# Evaluate on test dataset
loss, acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {acc*100:.2f}%")

# Plot training history
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss History")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy History")
plt.legend()
plt.show()

# Decode review text
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, "?") for i in encoded_review if i >= 3])

# Example prediction
sample_text = x_test[0].reshape(1, -1)
pred = model.predict(sample_text)
print("Review text:\\n", decode_review(x_test[0]))
print("Predicted Sentiment:", "Positive" if pred[0][0]>0.5 else "Negative")
print("Actual Sentiment:", "Positive" if y_test[0]==1 else "Negative")
  `,

  algorithm: `
1. Load IMDB dataset with top 10,000 words
2. Pad sequences to fixed length (max_len=200)
3. Define RNN model with embedding, SimpleRNN, and Dense layers
4. Compile model with binary cross-entropy loss and Adam optimizer
5. Train model on training data with validation split
6. Evaluate model on test data
7. Plot loss and accuracy curves
8. Decode a sample review from word indices
9. Predict sentiment for sample review and compare with actual label
  `,

  viva: [
    {
      q: "Why use RNNs instead of Bag-of-Words for sentiment analysis?",
      a: "RNNs consider the order of words and maintain context via hidden states. Bag-of-Words ignores word order, so it cannot capture negations or sequence-dependent meanings."
    },
    {
      q: "What does the Embedding layer do?",
      a: "The Embedding layer maps each word index to a dense vector representation, allowing the model to learn semantic relationships between words."
    },
    {
      q: "Why do we pad sequences?",
      a: "Padding ensures all input sequences have the same length, which is required for batch processing in neural networks."
    },
    {
      q: "What loss function is used and why?",
      a: "Binary Cross-Entropy is used because sentiment classification is a binary problem (positive/negative)."
    }
  ]
};
