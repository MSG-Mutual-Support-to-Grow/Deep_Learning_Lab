# Exercise 2: Regression - House Price Prediction

## Aim
To implement a fully connected neural network for a regression task to predict house prices using the California Housing dataset.

## Description
This exercise builds a regression model using a fully connected neural network to predict continuous values (house prices) based on various features such as location, number of rooms, population, and other housing characteristics. Unlike classification, regression predicts numerical values rather than discrete classes.

## Dataset
**California Housing Dataset**
- **Samples**: 20,640 house records
- **Features**: 8 numerical features
  - MedInc: Median income in block group
  - HouseAge: Median house age in block group
  - AveRooms: Average number of rooms per household
  - AveBedrms: Average number of bedrooms per household
  - Population: Block group population
  - AveOccup: Average number of household members
  - Latitude: Block group latitude
  - Longitude: Block group longitude
- **Target**: Median house value (in $100,000s)

## Model Architecture

### Network Structure
```
Input Layer: 8 features (scaled)
    ↓
Hidden Layer 1: 64 neurons, ReLU activation
    ↓
Hidden Layer 2: 32 neurons, ReLU activation
    ↓
Output Layer: 1 neuron (continuous value)
```

### Key Components
- **Input**: 8 standardized features
- **Hidden Layers**:
  - Layer 1: 64 neurons with ReLU activation
  - Layer 2: 32 neurons with ReLU activation
- **Output Layer**: 1 neuron (no activation for regression)
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam (learning_rate=0.001)
- **Metrics**: Mean Absolute Error (MAE)

## Procedure

### Step 1: Import Required Libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.datasets import fetch_california_housing
```

### Step 2: Load and Explore Data
```python
# Load California Housing dataset
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

print("Shape of X:", X.shape)  # (20640, 8)
print("Shape of y:", y.shape)  # (20640,)
```

### Step 3: Preprocess Data
```python
# Standardize features (important for neural networks)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and test sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
```

### Step 4: Build the Model
```python
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))  # Single output for regression
```

### Step 5: Compile the Model
```python
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',  # Mean Squared Error
    metrics=['mae']  # Mean Absolute Error
)
```

### Step 6: Train the Model
```python
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=8,
    batch_size=32,
    verbose=1
)
```

### Step 7: Evaluate the Model
```python
loss, mae = model.evaluate(X_test, y_test)
print(f"Test Loss (MSE): {loss:.4f}")
print(f"Test MAE: {mae:.4f}")
```

### Step 8: Visualize Training Progress
```python
plt.figure(figsize=(12, 5))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.title('Loss over Epochs')
plt.legend()

# Plot MAE
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.title('MAE over Epochs')
plt.legend()

plt.show()
```

### Step 9: Make Predictions
```python
y_pred = model.predict(X_test)

# Display sample predictions
for i in range(5):
    print(f"Predicted: {y_pred[i][0]:.2f}, Actual: {y_test.values[i]:.2f}")
```

## Expected Output

### Training Progress
```
Epoch 1/8
413/413 [==============================] - 2s 3ms/step - loss: 1.3542 - mae: 0.8234 - val_loss: 0.5823 - val_mae: 0.5621
Epoch 2/8
413/413 [==============================] - 1s 2ms/step - loss: 0.5234 - mae: 0.5387 - val_loss: 0.4912 - val_mae: 0.5123
...
Epoch 8/8
413/413 [==============================] - 1s 2ms/step - loss: 0.3156 - mae: 0.4012 - val_loss: 0.3923 - val_mae: 0.4356
```

### Test Results
```
129/129 [==============================] - 0s 2ms/step - loss: 0.3856 - mae: 0.4289
Test Loss (MSE): 0.3856
Test MAE: 0.4289
```

### Sample Predictions
```
Predicted: 2.15, Actual: 2.05
Predicted: 1.78, Actual: 1.95
Predicted: 3.42, Actual: 3.67
Predicted: 1.23, Actual: 1.18
Predicted: 2.89, Actual: 2.74
```

## Interpretation of Results

1. **Mean Absolute Error (MAE)**: ~0.43 means predictions are off by approximately $43,000 on average (since target is in $100,000s)

2. **Mean Squared Error (MSE)**: ~0.39 indicates the average squared difference between predicted and actual values

3. **Loss Curves**: 
   - Both training and validation losses decrease over epochs
   - Validation loss stabilizes, indicating good generalization
   - No significant overfitting observed

4. **Predictions**: Most predictions are close to actual values, demonstrating the model's effectiveness

## Key Learnings

1. **Feature Scaling**: StandardScaler is crucial for neural networks as it normalizes features to similar ranges

2. **Regression vs Classification**: 
   - Output layer has 1 neuron (not softmax)
   - Loss function is MSE (not categorical crossentropy)
   - Predictions are continuous values

3. **Evaluation Metrics**:
   - MSE penalizes larger errors more heavily
   - MAE is more interpretable (average error in original units)

4. **Architecture**: Deeper networks with more neurons can capture complex relationships in housing data

## Visualizations
The notebook generates two plots:
1. **Loss over Epochs**: Shows MSE for training and validation sets
2. **MAE over Epochs**: Shows mean absolute error progression

## Files
- `DL_EX2.ipynb`: Jupyter notebook containing the implementation

## Requirements
```
tensorflow>=2.0
numpy
pandas
scikit-learn
matplotlib
```

## How to Run
1. Open `DL_EX2.ipynb` in Jupyter Notebook or Google Colab
2. Run all cells sequentially
3. Observe training progress, evaluation metrics, and visualizations

## Conclusion
This exercise demonstrates how fully connected neural networks can effectively solve regression problems. The model learns complex non-linear relationships between housing features and prices, achieving reasonable prediction accuracy. Feature scaling and appropriate loss functions (MSE) are key to successful regression with neural networks.
