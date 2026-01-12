import numpy as np
import sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Simple callback - prints every 100 batches
class PrintProgress(Callback):
    def on_batch_end(self, batch, logs=None):
        if batch % 100 == 0:
            print(f"Batch {batch} - accuracy: {logs['accuracy']:.4f} - loss: {logs['loss']:.4f}")

# Parse arguments
npz_file = sys.argv[1]
mode = sys.argv[2]  # "train" or "test"

# Load data
print(f"Loading data from {npz_file}...")
data = np.load(npz_file)
X = data["images"]
y = data["labels"]

# Reshape and normalize
X = X.reshape(-1, 28, 28, 1).astype('float32') / 255.0
print(f"Data shape: {X.shape}, Labels shape: {y.shape}")

# Build model architecture
def create_model():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

if mode == "train":
    print("\n=== TRAINING MODE ===")
    
    # Split data 80/20
    N = X.shape[0]
    split = int(0.8 * N)
    indices = np.random.permutation(N)
    X_train, X_test = X[indices[:split]], X[indices[split:]]
    y_train, y_test = y[indices[:split]], y[indices[split:]]
    
    print(f"Train set: {X_train.shape[0]} images")
    print(f"Test set: {X_test.shape[0]} images")
    
    # Create and train model
    model = create_model()
    
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train, 
        batch_size=32, 
        epochs=4, 
        validation_split=0.1, 
        verbose=1
    )
    
    # Save model weights
    model.save_weights('model_weights.weights.h5')
    print("\nModel weights saved to 'model_weights.weights.h5'")
    
    # Evaluate on test set
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {acc:.4f}")
    
    # Predictions for confusion matrix
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix (Accuracy: {acc:.4f})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix_train.png')
    print("Confusion matrix saved to 'confusion_matrix_train.png'")
    plt.show()

elif mode == "test":
    print("\n=== TEST MODE ===")
    
    # Create model and load weights
    model = create_model()
    try:
        model.load_weights('model_weights.weights.h5')
        print("Model weights loaded from 'model_weights.weights.h5'")
    except:
        print("ERROR: Could not load model weights. Please train the model first.")
        sys.exit(1)
    
    # Evaluate on test data
    loss, acc = model.evaluate(X, y, verbose=0)
    print(f"\nTest accuracy: {acc:.4f}")
    
    # Predictions for confusion matrix
    y_pred = np.argmax(model.predict(X, verbose=0), axis=1)
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - Test Set (Accuracy: {acc:.4f})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix_test.png')
    print("Confusion matrix saved to 'confusion_matrix_test.png'")
    plt.show()

else:
    print(f"ERROR: Invalid mode '{mode}'. Use 'train' or 'test'")
    sys.exit(1)