# Exercise 1: Multiclass Classification

## Aim
To build and train fully connected neural networks for multiclass classification problems using the MNIST and Fashion MNIST datasets.

## Description
This exercise implements two separate neural network models:
1. **MNIST Classification**: Classifies handwritten digits (0-9) from the MNIST dataset
2. **Fashion MNIST Classification**: Classifies clothing items from the Fashion MNIST dataset into 10 categories

Both models use a simple fully connected architecture with dense layers to demonstrate the fundamentals of multiclass classification in deep learning.

## Datasets
- **MNIST**: 70,000 grayscale images (28x28 pixels) of handwritten digits
  - Training set: 60,000 images
  - Test set: 10,000 images
  - Classes: 10 (digits 0-9)

- **Fashion MNIST**: 70,000 grayscale images (28x28 pixels) of clothing items
  - Training set: 60,000 images
  - Test set: 10,000 images
  - Classes: 10 (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)

## Model Architecture

### Network Structure
```
Input Layer: 28x28 pixels (flattened to 784 features)
    ↓
Hidden Layer 1: 128 neurons, ReLU activation
    ↓
Hidden Layer 2: 64 neurons, ReLU activation
    ↓
Output Layer: 10 neurons, Softmax activation
```

### Key Components
- **Input**: Flattened 28x28 grayscale images (784 features)
- **Hidden Layers**: 
  - Layer 1: 128 neurons with ReLU activation
  - Layer 2: 64 neurons with ReLU activation
- **Output Layer**: 10 neurons with Softmax activation (one per class)
- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam
- **Metrics**: Accuracy

## Procedure

### Step 1: Import Required Libraries
```python
import tensorflow as tf
from tensorflow.keras.datasets import mnist, fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
```

### Step 2: Load and Preprocess Data
```python
# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to range [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
```

### Step 3: Build the Model
```python
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
```

### Step 4: Compile the Model
```python
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

### Step 5: Train the Model
```python
model.fit(
    x_train, y_train, 
    epochs=10, 
    batch_size=64, 
    validation_split=0.1
)
```

### Step 6: Evaluate the Model
```python
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
```

### Step 7: Repeat for Fashion MNIST
Replace `mnist.load_data()` with `fashion_mnist.load_data()` and follow the same steps.

## Expected Output

### MNIST Results
```
Epoch 1/10
844/844 [==============================] - 3s 3ms/step - loss: 0.2613 - accuracy: 0.9241 - val_loss: 0.1352 - val_accuracy: 0.9597
Epoch 2/10
844/844 [==============================] - 2s 3ms/step - loss: 0.1141 - accuracy: 0.9657 - val_loss: 0.1036 - val_accuracy: 0.9693
...
Epoch 10/10
844/844 [==============================] - 2s 3ms/step - loss: 0.0234 - accuracy: 0.9925 - val_loss: 0.0891 - val_accuracy: 0.9775

Test Accuracy: 0.9756
```

### Fashion MNIST Results
```
Epoch 1/10
844/844 [==============================] - 3s 3ms/step - loss: 0.5123 - accuracy: 0.8172 - val_loss: 0.4214 - val_accuracy: 0.8467
Epoch 2/10
844/844 [==============================] - 2s 3ms/step - loss: 0.3823 - accuracy: 0.8613 - val_loss: 0.3891 - val_accuracy: 0.8583
...
Epoch 10/10
844/844 [==============================] - 2s 3ms/step - loss: 0.2134 - accuracy: 0.9213 - val_loss: 0.3421 - val_accuracy: 0.8817

Test Accuracy: 0.8792
```

## Key Observations

1. **MNIST Performance**: The model achieves ~97-98% accuracy on MNIST, which is expected for this relatively simple dataset with a basic fully connected network.

2. **Fashion MNIST Performance**: The model achieves ~87-88% accuracy on Fashion MNIST, which is lower than MNIST because clothing items have more complex patterns and variations.

3. **Training Behavior**: 
   - Loss decreases steadily over epochs
   - Validation accuracy improves and stabilizes
   - Some overfitting may occur in later epochs (train accuracy > validation accuracy)

4. **One-Hot Encoding**: Labels are converted to categorical format for compatibility with softmax output and categorical crossentropy loss.

## Files
- `DL_EX1.ipynb`: Jupyter notebook containing the implementation

## Requirements
```
tensorflow>=2.0
numpy
```

## How to Run
1. Open `DL_EX1.ipynb` in Jupyter Notebook or Google Colab
2. Run all cells sequentially
3. Observe training progress and final test accuracy for both datasets

## Conclusion
This exercise demonstrates the effectiveness of simple fully connected neural networks for multiclass image classification. While MNIST achieves high accuracy with minimal complexity, Fashion MNIST presents a more challenging task that may benefit from convolutional layers in more advanced exercises.
