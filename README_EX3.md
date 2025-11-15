# Exercise 3: Convolutional Neural Network (CNN) - CIFAR-10 Classification

## Aim
To construct and train a Convolutional Neural Network (CNN) for image classification on the CIFAR-10 dataset.

## Description
This exercise implements a CNN architecture to classify color images from the CIFAR-10 dataset into 10 different categories. Unlike fully connected networks, CNNs use convolutional layers that are specifically designed to process grid-like data such as images, making them highly effective for computer vision tasks.

## Dataset
**CIFAR-10**
- **Total Images**: 60,000 color images (32x32 pixels, RGB)
- **Training Set**: 50,000 images
- **Test Set**: 10,000 images
- **Classes**: 10 categories
  1. Airplane
  2. Automobile
  3. Bird
  4. Cat
  5. Deer
  6. Dog
  7. Frog
  8. Horse
  9. Ship
  10. Truck
- **Image Format**: 32x32x3 (height × width × RGB channels)

## Model Architecture

### Network Structure
```
Input: 32x32x3 RGB Image
    ↓
Conv2D Layer 1: 32 filters, 3x3 kernel, ReLU
    ↓
MaxPooling2D: 2x2 pool
    ↓
Conv2D Layer 2: 64 filters, 3x3 kernel, ReLU
    ↓
MaxPooling2D: 2x2 pool
    ↓
Conv2D Layer 3: 64 filters, 3x3 kernel, ReLU
    ↓
Flatten
    ↓
Dense Layer: 64 neurons, ReLU
    ↓
Dropout: 0.5
    ↓
Output Layer: 10 neurons, Softmax
```

### Key Components
- **Convolutional Layers**: Extract spatial features from images
  - Layer 1: 32 filters (3×3), ReLU activation
  - Layer 2: 64 filters (3×3), ReLU activation
  - Layer 3: 64 filters (3×3), ReLU activation
- **Pooling Layers**: Reduce spatial dimensions (2×2 max pooling)
- **Flatten Layer**: Convert 2D feature maps to 1D vector
- **Dense Layer**: 64 neurons with ReLU activation
- **Dropout**: 0.5 (50% dropout for regularization)
- **Output Layer**: 10 neurons with Softmax activation
- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam
- **Metrics**: Accuracy

## Procedure

### Step 1: Import Required Libraries
```python
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
```

### Step 2: Load and Preprocess Data
```python
# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

# Normalize pixel values to range [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0

# One-hot encode labels
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
```

### Step 3: Build the CNN Model
```python
model = models.Sequential()

# Convolutional layers
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Dense layers
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))
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
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=64,
    validation_data=(x_test, y_test)
)
```

### Step 6: Evaluate the Model
```python
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc * 100:.2f}%")
```

### Step 7: Visualize Training Progress
```python
plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()

plt.show()
```

## Expected Output

### Training Progress
```
Epoch 1/10
782/782 [==============================] - 45s 57ms/step - loss: 1.4523 - accuracy: 0.4712 - val_loss: 1.1234 - val_accuracy: 0.6023
Epoch 2/10
782/782 [==============================] - 43s 55ms/step - loss: 1.0234 - accuracy: 0.6387 - val_loss: 0.9823 - val_accuracy: 0.6587
Epoch 3/10
782/782 [==============================] - 42s 54ms/step - loss: 0.8745 - accuracy: 0.6934 - val_loss: 0.8912 - val_accuracy: 0.6923
...
Epoch 10/10
782/782 [==============================] - 41s 53ms/step - loss: 0.5234 - accuracy: 0.8156 - val_loss: 0.7823 - val_accuracy: 0.7345
```

### Final Test Results
```
313/313 - 3s - loss: 0.7823 - accuracy: 0.7345
Test accuracy: 73.45%
```

## CNN Architecture Advantages

1. **Spatial Feature Extraction**: Convolutional layers learn local patterns (edges, textures, shapes)

2. **Parameter Efficiency**: Shared weights across the image reduce parameters compared to fully connected networks

3. **Translation Invariance**: Pooling layers make the network robust to small translations

4. **Hierarchical Learning**: 
   - Early layers detect simple features (edges)
   - Middle layers detect complex patterns (textures)
   - Deep layers detect high-level features (objects)

## Key Observations

1. **Performance**: CNN achieves ~70-75% accuracy on CIFAR-10, significantly better than fully connected networks

2. **Training Time**: CNNs take longer to train due to convolution operations, but are more accurate

3. **Overfitting Prevention**:
   - Dropout (0.5) helps prevent overfitting
   - Validation accuracy tracks training accuracy reasonably well

4. **Visualization**: 
   - Training and validation curves show learning progress
   - Gap between train and validation indicates some overfitting

## Hyperparameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Epochs | 10 | Number of complete passes through training data |
| Batch Size | 64 | Number of samples per gradient update |
| Dropout Rate | 0.5 | Fraction of neurons randomly dropped |
| Learning Rate | Default (Adam) | Step size for weight updates |
| Filters | 32, 64, 64 | Number of convolutional filters per layer |

## Possible Improvements

1. **Data Augmentation**: Random flips, rotations, crops
2. **Deeper Networks**: Add more convolutional layers
3. **Batch Normalization**: Normalize layer inputs
4. **Learning Rate Scheduling**: Reduce learning rate over time
5. **Transfer Learning**: Use pre-trained models

## Files
- `DL_EX3.ipynb`: Jupyter notebook containing the implementation

## Requirements
```
tensorflow>=2.0
numpy
matplotlib
```

## How to Run
1. Open `DL_EX3.ipynb` in Jupyter Notebook or Google Colab
2. Run all cells sequentially
3. Observe training progress, accuracy curves, and final test accuracy

## Conclusion
This exercise demonstrates the power of Convolutional Neural Networks for image classification. CNNs automatically learn hierarchical feature representations from raw pixels, making them highly effective for computer vision tasks. The CIFAR-10 dataset presents a challenging problem with diverse object categories, and the CNN architecture achieves respectable accuracy through spatial feature extraction and pooling operations.
