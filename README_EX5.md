# Exercise 5: Recurrent Neural Networks (RNN) - Time Series Prediction

## Aim
To implement Recurrent Neural Networks (RNN) for time series prediction tasks including:
1. Daily minimum temperature prediction
2. Car sales forecasting
3. Stock price prediction

## Description
This exercise demonstrates the application of Simple RNN architecture for sequence prediction tasks. Unlike feedforward neural networks, RNNs maintain internal state (memory) that allows them to process sequential data and capture temporal dependencies. This makes them ideal for time series forecasting, where past values influence future predictions.

## What are RNNs?

Recurrent Neural Networks are designed to work with sequential data by:
- **Maintaining Memory**: Hidden states carry information from previous time steps
- **Processing Sequences**: Handle variable-length input sequences
- **Temporal Dependencies**: Learn patterns that evolve over time

### RNN Architecture
```
Input Sequence: [x₁, x₂, x₃, ..., xₜ]
    ↓
RNN Layer: Processes sequence step-by-step
    ↓ (hidden state updated at each step)
Output: yₜ₊₁ (next time step prediction)
```

## Task 1: Daily Minimum Temperature Prediction

### Dataset
- **Source**: `daily-minimum-temperatures-in-me.csv`
- **Features**: Date, Daily minimum temperature
- **Task**: Predict next day's minimum temperature
- **Sequence Length**: 10 days (use 10 previous days to predict the next)

### Model Architecture
```
Input: (batch_size, 10, 1) - 10 time steps
    ↓
SimpleRNN: 64 units, ReLU activation
    ↓
Dense: 1 unit (temperature prediction)
```

### Procedure

#### Step 1: Import Libraries and Load Data
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Load dataset
data = pd.read_csv("daily-minimum-temperatures-in-me.csv")
data.columns = data.columns.str.strip()
data['Date'] = pd.to_datetime(data['Date'])

# Clean numeric column (remove ? and convert to float)
data['Daily minimum temperatures'] = (
    data['Daily minimum temperatures']
    .astype(str)
    .str.replace('?', '', regex=False)
    .astype(float)
)
```

#### Step 2: Normalize Data
```python
values = data['Daily minimum temperatures'].values.reshape(-1, 1)

# Normalize to [0, 1] range
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
```

#### Step 3: Create Sequences
```python
X, y = [], []
time_step = 10

for i in range(len(scaled) - time_step):
    X.append(scaled[i:i + time_step])
    y.append(scaled[i + time_step])

X, y = np.array(X), np.array(y)
```

#### Step 4: Build and Train RNN
```python
model = Sequential([
    SimpleRNN(64, activation='relu', input_shape=(time_step, 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=20, batch_size=16, verbose=1)
```

#### Step 5: Predict and Visualize
```python
predicted = model.predict(X)
predicted = scaler.inverse_transform(predicted)
real = scaler.inverse_transform(y)

plt.figure(figsize=(10, 6))
plt.plot(real, label='Actual', color='blue')
plt.plot(predicted, label='Predicted', color='red')
plt.legend()
plt.title('Daily Minimum Temperature Prediction using RNN')
plt.xlabel('Days')
plt.ylabel('Temperature (°C)')
plt.show()
```

### Expected Output
```
Epoch 1/20
235/235 [==============================] - 3s 10ms/step - loss: 0.0234
Epoch 2/20
235/235 [==============================] - 2s 9ms/step - loss: 0.0156
...
Epoch 20/20
235/235 [==============================] - 2s 9ms/step - loss: 0.0045
```

## Task 2: Car Sales Prediction

### Dataset
- **Source**: `Car_sales.csv`
- **Features**: Sales_in_thousands (monthly car sales)
- **Task**: Forecast future car sales
- **Sequence Length**: 12 months (yearly pattern)

### Model Architecture
```
Input: (batch_size, 12, 1) - 12 time steps
    ↓
SimpleRNN: 100 units, tanh activation
    ↓
Dense: 1 unit (sales prediction)
```

### Procedure

#### Key Steps
```python
# Load and preprocess
data = pd.read_csv("Car_sales.csv")
sales = data['Sales_in_thousands'].astype(float).values.reshape(-1, 1)

# Normalize
scaler = MinMaxScaler(feature_range=(0, 1))
sales_scaled = scaler.fit_transform(sales)

# Create sequences (12-month lookback)
X, y = [], []
time_step = 12
for i in range(len(sales_scaled) - time_step):
    X.append(sales_scaled[i:i + time_step])
    y.append(sales_scaled[i + time_step])

X, y = np.array(X), np.array(y)

# Build model
model = Sequential([
    SimpleRNN(100, activation='tanh', input_shape=(time_step, 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=30, batch_size=8, verbose=1)
```

### Expected Output
```
Epoch 1/30
52/52 [==============================] - 2s 25ms/step - loss: 0.0523
Epoch 2/30
52/52 [==============================] - 1s 22ms/step - loss: 0.0312
...
Epoch 30/30
52/52 [==============================] - 1s 21ms/step - loss: 0.0089
```

## Task 3: Stock Price Prediction

### Dataset
- **Source**: `stock_prices.csv`
- **Features**: Date, Close (closing stock price)
- **Task**: Predict next day's closing price
- **Sequence Length**: 30 days

### Model Architecture
```
Input: (batch_size, 30, 1) - 30 time steps
    ↓
SimpleRNN: 80 units, ReLU activation
    ↓
Dense: 1 unit (price prediction)
```

### Procedure

#### Key Steps
```python
# Load and sort data
data = pd.read_csv("stock_prices.csv")
data['date'] = pd.to_datetime(data['date'])
data = data.sort_values('date')

prices = data['close'].values.reshape(-1, 1)

# Normalize
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)

# Create sequences (30-day lookback)
time_step = 30
X, y = [], []
for i in range(time_step, len(scaled_prices)):
    X.append(scaled_prices[i-time_step:i, 0])
    y.append(scaled_prices[i, 0])

X, y = np.array(X), np.array(y)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Build model
model = Sequential([
    SimpleRNN(80, activation='relu', input_shape=(time_step, 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=10, batch_size=16, verbose=1)
```

### Expected Output
```
Epoch 1/10
48/48 [==============================] - 2s 28ms/step - loss: 0.0189
Epoch 2/10
48/48 [==============================] - 1s 24ms/step - loss: 0.0098
...
Epoch 10/10
48/48 [==============================] - 1s 23ms/step - loss: 0.0034
```

## Key Concepts

### 1. Sequence Creation
```python
for i in range(len(data) - time_step):
    X.append(data[i:i + time_step])  # Past values
    y.append(data[i + time_step])    # Future value
```
- Creates sliding windows over time series
- `time_step`: Number of past observations used for prediction

### 2. Data Normalization
```python
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
```
- Scales values to [0, 1] range
- Helps neural network training stability
- Must inverse transform predictions for interpretation

### 3. RNN Layer
```python
SimpleRNN(units, activation='relu', input_shape=(time_step, 1))
```
- `units`: Number of RNN neurons (memory capacity)
- `activation`: Non-linear activation function
- Maintains hidden state across time steps

## Common Hyperparameters

| Parameter | Temperature | Car Sales | Stock Price |
|-----------|-------------|-----------|-------------|
| Time Steps | 10 | 12 | 30 |
| RNN Units | 64 | 100 | 80 |
| Activation | ReLU | tanh | ReLU |
| Epochs | 20 | 30 | 10 |
| Batch Size | 16 | 8 | 16 |

## Evaluation Metrics

### Mean Squared Error (MSE)
- Primary loss function
- Measures average squared difference between predictions and actual values
- Lower is better

### Visual Inspection
- Plot actual vs predicted values
- Check if model captures trends and patterns
- Identify systematic errors or biases

## Limitations of Simple RNN

1. **Vanishing Gradient**: Difficulty learning long-term dependencies
2. **Short Memory**: Forgets information from distant past
3. **Training Instability**: Can be difficult to train on very long sequences

**Solution**: Use LSTM or GRU for better long-term memory (covered in Exercise 6)

## Visualizations
All three tasks generate plots showing:
- **Blue Line**: Actual values
- **Red Line**: Predicted values
- Comparison shows model's forecasting accuracy

## Files
- `DL_EX5.ipynb`: Jupyter notebook containing all three implementations
- `daily-minimum-temperatures-in-me.csv`: Temperature dataset
- `Car_sales.csv`: Car sales dataset
- `stock_prices.csv`: Stock price dataset

## Requirements
```
tensorflow>=2.0
numpy
pandas
matplotlib
scikit-learn
```

## How to Run
1. Ensure all CSV files are in the working directory
2. Open `DL_EX5.ipynb` in Jupyter Notebook or Google Colab
3. Run cells sequentially for each task
4. Observe training progress and prediction plots

## Conclusion
This exercise demonstrates RNN's effectiveness for time series prediction across three different domains:
- **Temperature**: Natural cyclic patterns
- **Car Sales**: Business/seasonal trends
- **Stock Prices**: Financial time series with noise

RNNs capture temporal dependencies and make reasonable predictions, though more advanced architectures like LSTM can further improve performance for complex sequences with long-term dependencies.
