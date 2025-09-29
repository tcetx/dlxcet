export default {
  name: "EXP 2: Fully Connected Deep Neural Network for Classification",
  theory: `In this experiment, the classification task is performed on the Iris dataset.

Input features (X): 4 numeric features per flower sample
- sepal length
- sepal width
- petal length
- petal width

Target (y): the species of iris flower
0 → setosa
1 → versicolor
2 → virginica

The deep neural network learns to classify a flower into one of the three species based on its measurements.`,
  
  code: `
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

# One-hot encode target labels
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Build the model
model = Sequential([
    Dense(16, input_shape=(X.shape[1],), activation='relu'),
    Dense(12, activation='relu'),
    Dense(3, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                    epochs=50, batch_size=8, verbose=1)

# Evaluate the model
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {acc*100:.2f}%")

# Predictions
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot using seaborn heatmap
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Plot training history
plt.figure(figsize=(12,5))

# Loss plot
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Model Loss")
plt.legend()

# Accuracy plot
plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Model Accuracy")
plt.legend()
plt.show()
  `,

  algorithm: `
1. Load Iris dataset and preprocess it (scaling + one-hot encoding)
2. Split dataset into training and testing sets
3. Build a Sequential neural network with 2 hidden layers
4. Compile the model with optimizer and loss
5. Train the model on the training set
6. Evaluate the model on the test set
7. Make predictions and compute confusion matrix
8. Plot training history and confusion matrix
  `,
  
  viva: [
    {
      q: "Why do we use multiple hidden layers in a deep neural network?",
      a: "Multiple hidden layers allow a neural network to learn hierarchical representations of data. The first layers capture low-level features, while deeper layers capture more abstract concepts. This increases the model's ability to approximate complex non-linear functions, improving performance for tasks like classification. Each layer applies transformations on its input, enabling the network to model interactions between features that a shallow network might miss. Using multiple layers also allows for feature reuse and reduces the need for manual feature engineering. In the Iris classification task, multiple layers help the network distinguish subtle differences between the flower species based on their measurements."
    },
    {
      q: "What role does softmax activation play in classification tasks?",
      a: "Softmax activation is used in the output layer for multi-class classification. It converts raw scores (logits) into probabilities that sum to 1, making them interpretable as class likelihoods. Each output neuron corresponds to a class, and softmax emphasizes the largest logit while suppressing smaller ones, effectively 'choosing' the most likely class. In the Iris dataset example, softmax produces three probabilities representing the likelihood of a sample belonging to setosa, versicolor, or virginica. This allows for easy prediction by selecting the class with the highest probability and enables the use of categorical cross-entropy loss for training."
    },
    {
      q: "Why is feature scaling important when training neural networks?",
      a: "Feature scaling standardizes the range of input variables, ensuring that all features contribute equally to the model's learning. Neural networks are sensitive to the scale of input data because large differences can lead to slow convergence or unstable training. Techniques like standardization (zero mean, unit variance) help in faster and more stable training, preventing some weights from dominating the learning process. In the Iris dataset, features like petal length and sepal width have different ranges; scaling them ensures that the network learns efficiently from all features, resulting in better accuracy and generalization."
    }
  ]
};
