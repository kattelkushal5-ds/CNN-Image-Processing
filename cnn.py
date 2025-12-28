import numpy as np
import sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import Callback

# Simple callback - prints every 100 batches
class PrintProgress(Callback):
    def on_batch_end(self, batch, logs=None):
        if batch % 100 == 0:
            print(f"Batch {batch} - accuracy: {logs['accuracy']:.4f} - loss: {logs['loss']:.4f}")

# Load data
data = np.load(sys.argv[1])
X = data["images"]
y = data["labels"]

# Reshape and normalize
X = X.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Split data
N = X.shape[0]
split = int(0.8 * N)
indices = np.random.permutation(N)

X_train, X_test = X[indices[:split]], X[indices[split:]]
y_train, y_test = y[indices[:split]], y[indices[split:]]

# Build model
model = Sequential([
    Conv2D(8, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train with custom printing (verbose=0 turns off default progress bar)
model.fit(X_train, y_train, batch_size=8, epochs=5, validation_split=0.1, 
          verbose=0, callbacks=[PrintProgress()])

# Test
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy: {acc:.4f}")