import numpy as np
import sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
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
if len(sys.argv) < 3:
    print("Usage: python3 cnn.py <npz_file> <mode> [model_type]")
    print("  mode: 'train' or 'test'")
    print("  model_type (optional): 'dnn' or 'cnn' (default: 'cnn')")
    sys.exit(1)

npz_file = sys.argv[1]
mode = sys.argv[2]  # "train" or "test"
model_type = sys.argv[3] if len(sys.argv) > 3 else "cnn"  # "dnn" or "cnn"

if model_type not in ["dnn", "cnn"]:
    print(f"ERROR: Invalid model_type '{model_type}'. Use 'dnn' or 'cnn'")
    sys.exit(1)

# Load data
print(f"Loading data from {npz_file}...")
data = np.load(npz_file)
X = data["images"]
y = data["labels"]

# Reshape and normalize
if model_type == "cnn":
    X = X.reshape(-1, 28, 28, 1).astype('float32') / 255.0  # CNN expects 4D input ---(batch_size, height, width, channels)
    print(f"Data shape (CNN): {X.shape}, Labels shape: {y.shape}")
else:  # dnn
    X = X.reshape(-1, 28 * 28).astype('float32') / 255.0  # DNN expects flattened input ---(batch_size, features) (N, 28 * 28)
    print(f"Data shape (DNN): {X.shape}, Labels shape: {y.shape}")

# Build model architecture
def create_dnn_model():
    """Dense Neural Network with fully connected layers"""
    model = Sequential([
        Dense(256, activation='relu', input_shape=(784,)),
        Dropout(0.3), #30% of neurons -0
        Dense(128, activation='relu'), #Input: 256 → output 128
        Dropout(0.3),
        Dense(64, activation='relu'), #Input: 128 → output 64
        Dropout(0.2),
        Dense(10, activation='softmax') #10 neurons for 10 classes
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def create_cnn_model():
    """Convolutional Neural Network (LeNet-5 style)"""
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),   # 32 filters, output (26,26,32) → convolution reduces size (28-3+1)
        MaxPooling2D((2,2)), #Output (13,13,32)
        Conv2D(64, (3,3), activation='relu'), #(11,11,64)
        MaxPooling2D((2,2)), #(5,5,64)
        Flatten(), #(5,5,64) → 1600
        Dense(128, activation='relu'),  #1600 -- 128
        Dropout(0.3),
        Dense(10, activation='softmax') # 10
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def create_model():
    """Create model based on selected type"""
    if model_type == "dnn":
        print(f"\nBuilding DNN model...")
        return create_dnn_model()
    else:
        print(f"\nBuilding CNN model...")
        return create_cnn_model()



# Define weight file names based on model type
weights_file = f'model_weights_{model_type}.weights.h5'



if mode == "train":
    print(f"\n=== TRAINING MODE ({model_type.upper()}) ===")
    
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
    model.summary()
    
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train, 
        batch_size=32, 
        epochs=15, 
        validation_split=0.1, 
        verbose=1
    )
    
    # Save model weights
    model.save_weights(weights_file)
    print(f"\nModel weights saved to '{weights_file}'")
    
    # Evaluate on test set
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {acc:.4f}")
    
    # Predictions for confusion matrix
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_type.upper()} (Accuracy: {acc:.4f})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_file = f'confusion_matrix_train_{model_type}.png'
    plt.savefig(cm_file)
    print(f"Confusion matrix saved to '{cm_file}'")
    plt.show()

elif mode == "test":
    print(f"\n=== TEST MODE ({model_type.upper()}) ===")
    
    # Create model and load weights
    model = create_model()
    try:
        model.load_weights(weights_file)
        print(f"Model weights loaded from '{weights_file}'")
    except:
        print(f"ERROR: Could not load model weights from '{weights_file}'.")
        print(f"Please train the {model_type.upper()} model first using:")
        print(f"  python3 cnn.py train.npz train {model_type}")
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
    plt.title(f'Confusion Matrix - Test Set - {model_type.upper()} (Accuracy: {acc:.4f})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_file = f'confusion_matrix_test_{model_type}.png'
    plt.savefig(cm_file)
    print(f"Confusion matrix saved to '{cm_file}'")
    plt.show()

else:
    print(f"ERROR: Invalid mode '{mode}'. Use 'train' or 'test'")
    sys.exit(1)