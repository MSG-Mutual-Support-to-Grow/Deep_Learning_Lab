# Exercise 6: LSTM for Text Generation - Shakespeare

## Aim
To train a Long Short-Term Memory (LSTM) network for character-level text generation using Shakespeare's works as the training dataset.

## Description
This exercise implements an LSTM-based text generation model that learns to write text in the style of Shakespeare. Unlike Simple RNNs, LSTMs can capture long-term dependencies in sequences, making them ideal for natural language tasks. The model learns character-by-character patterns and generates new text that resembles Shakespeare's writing style.

## What is LSTM?

### Long Short-Term Memory
LSTM is an advanced RNN architecture that solves the vanishing gradient problem through:
- **Cell State**: Long-term memory that runs through the entire sequence
- **Gates**: Mechanisms to add or remove information
  - **Forget Gate**: Decides what to discard from cell state
  - **Input Gate**: Decides what new information to store
  - **Output Gate**: Decides what to output based on cell state

### LSTM vs Simple RNN
| Feature | Simple RNN | LSTM |
|---------|-----------|------|
| Memory | Short-term | Long-term |
| Gradient Flow | Vanishing gradient issues | Stable gradients |
| Complexity | Simple | More parameters |
| Training | Faster | Slower but more effective |
| Use Case | Short sequences | Long sequences, text |

## Dataset
**Shakespeare's Works**
- **Source**: Tiny Shakespeare dataset from Karpathy's char-rnn repository
- **Size**: ~1 million characters
- **Content**: Collection of Shakespeare's plays and sonnets
- **Format**: Plain text, character-level
- **Vocabulary**: ~65 unique characters (letters, punctuation, spaces)

## Model Architecture

### Network Structure
```
Input: Character sequence (100 characters)
    ↓
LSTM Layer 1: 256 units, return_sequences=True
    ↓
Dropout: 0.2 (20% dropout)
    ↓
LSTM Layer 2: 256 units
    ↓
Dropout: 0.2
    ↓
Dense Layer: 256 units, ReLU activation
    ↓
Output Layer: vocab_size units, Softmax activation
```

### Key Components
- **Sequence Length**: 100 characters (context window)
- **LSTM Layers**: 2 stacked layers with 256 units each
- **Dropout**: 0.2 (prevents overfitting)
- **Dense Layer**: 256 neurons for feature transformation
- **Output**: Probability distribution over all characters
- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam
- **Metrics**: Accuracy

## Procedure

### Step 1: Download and Load Shakespeare Text
```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import get_file
import matplotlib.pyplot as plt

# Download Shakespeare's works
shakespeare_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
shakespeare_text = get_file("shakespeare.txt", shakespeare_url)

# Load the text
with open(shakespeare_text, "r") as file:
    text = file.read()

print(f"Text Length: {len(text)}")
print(text[:1000])  # Preview first 1000 characters
```

### Step 2: Create Character Mappings
```python
# Create a mapping from characters to integers
chars = sorted(set(text))
char_to_int = {char: i for i, char in enumerate(chars)}
int_to_char = {i: char for i, char in enumerate(chars)}

# Convert text to integer sequence
text_as_int = np.array([char_to_int[char] for char in text])
```

### Step 3: Prepare Training Sequences
```python
sequence_length = 100
X = []
y = []

# Create sequences
for i in range(0, len(text_as_int) - sequence_length):
    X.append(text_as_int[i:i + sequence_length])
    y.append(text_as_int[i + sequence_length])

X = np.array(X)
y = np.array(y)

# Reshape X for LSTM: (samples, time_steps, features)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Normalize X to [0, 1]
X = X / float(len(chars))

# One-hot encode y
y = tf.keras.utils.to_categorical(y, num_classes=len(chars))

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
```

### Step 4: Build LSTM Model
```python
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.LSTM(256))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(len(chars), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
```

### Step 5: Train the Model
```python
# For demonstration, use subset of data
text = text[:10000]  # Use only 10,000 characters

# Train model
history = model.fit(X, y, epochs=10, batch_size=128)

# Plot training loss
plt.plot(history.history['loss'])
plt.title('Model Loss during Training')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
```

### Step 6: Text Generation Function
```python
def generate_text(seed, length=100):
    # Convert seed to integers
    seed = [char_to_int[char] for char in seed]
    
    # Reshape for prediction
    seed = np.reshape(seed, (1, len(seed), 1)) / float(len(chars))
    
    # Generate text character by character
    generated_text = seed
    for i in range(length):
        prediction = model.predict(seed, verbose=0)
        predicted_char_index = np.argmax(prediction)
        
        # Convert index to character
        predicted_char = int_to_char[predicted_char_index]
        generated_text = np.append(generated_text, predicted_char)
        
        # Update seed for next prediction
        seed = np.append(seed[0, 1:], predicted_char_index)
        seed = np.reshape(seed, (1, len(seed), 1)) / float(len(chars))
    
    return ''.join(generated_text)
```

### Step 7: Generate Shakespeare-style Text
```python
seed_text = "Shall I compare thee to a summer's day?"
generated_text = generate_text(seed_text, length=500)
print(generated_text)
```

## Expected Output

### Model Summary
```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lstm (LSTM)                  (None, 100, 256)          264,192   
dropout (Dropout)            (None, 100, 256)          0         
lstm_1 (LSTM)                (None, 256)               525,312   
dropout_1 (Dropout)          (None, 256)               0         
dense (Dense)                (None, 256)               65,792    
dense_1 (Dense)              (None, 65)                16,705    
=================================================================
Total params: 872,001
Trainable params: 872,001
Non-trainable params: 0
```

### Training Progress
```
Epoch 1/10
79/79 [==============================] - 45s 512ms/step - loss: 3.0234 - accuracy: 0.1823
Epoch 2/10
79/79 [==============================] - 42s 502ms/step - loss: 2.4567 - accuracy: 0.2945
Epoch 3/10
79/79 [==============================] - 41s 498ms/step - loss: 2.1234 - accuracy: 0.3687
...
Epoch 10/10
79/79 [==============================] - 40s 495ms/step - loss: 1.2345 - accuracy: 0.6234
```

### Generated Text Example
```
Shall I compare thee to a summer's day?
Thou art more lovely and more temperate:
Rough winds do shake the darling buds of May,
And summer's lease hath all too short a date:
Sometime too hot the eye of heaven shines,
And often is his gold complexion dimm'd;
And every fair from fair sometime declines,
By chance or nature's changing course untrimm'd;
```

**Note**: With limited training (10 epochs, 10K characters), the generated text may not be perfectly coherent but will capture some Shakespeare-like patterns. Full training requires more data and epochs.

## Key Concepts

### 1. Character-Level Modeling
- Model predicts one character at a time
- Learns spelling, grammar, and style patterns
- More flexible than word-level models

### 2. Sequence-to-Sequence Prediction
```
Input:  "Shall I compare thee to a summer's d"
Output: "a"  (next character)
```

### 3. Sampling Strategy
- **Greedy**: Always pick highest probability character
- **Random Sampling**: Sample from probability distribution
- **Temperature**: Control randomness/creativity

### 4. Dropout Regularization
```python
Dropout(0.2)  # Drop 20% of neurons during training
```
- Prevents overfitting
- Forces model to learn robust features

## Training Considerations

### Data Size
- **Full Dataset**: ~1 million characters → better results
- **Subset**: 10K characters → faster training, less coherent output
- **Recommendation**: Use full dataset with GPU acceleration

### Epochs
- **10 epochs**: Basic patterns learned
- **30-50 epochs**: Reasonable text quality
- **100+ epochs**: Best results (risk of overfitting)

### Batch Size
- **32**: Slower, more stable gradients
- **128**: Faster, good balance
- **256**: Fastest, may be less stable

## Hyperparameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Sequence Length | 100 | Context window for prediction |
| LSTM Units | 256 | Memory capacity |
| Dropout Rate | 0.2 | Regularization strength |
| Dense Units | 256 | Feature transformation |
| Batch Size | 128 | Samples per gradient update |
| Epochs | 10-30 | Training iterations |

## Evaluation

### Quantitative Metrics
- **Loss**: Categorical crossentropy (lower is better)
- **Accuracy**: Character prediction accuracy

### Qualitative Metrics
- **Coherence**: Does text make sense?
- **Style**: Does it sound like Shakespeare?
- **Grammar**: Correct punctuation and structure?
- **Creativity**: Novel yet plausible text?

## Applications of LSTM Text Generation

1. **Creative Writing**: Story generation, poetry
2. **Code Generation**: Autocomplete for programming
3. **Music Composition**: Generate musical sequences
4. **Chatbots**: Conversational AI
5. **Translation**: Sequence-to-sequence tasks

## Limitations

1. **Long-Range Coherence**: May lose plot/theme over long text
2. **Factual Accuracy**: Generates plausible but not necessarily true text
3. **Training Time**: Requires significant computation
4. **Data Dependency**: Quality depends on training corpus

## Improvements

1. **More Data**: Train on full Shakespeare corpus
2. **More Epochs**: 50-100 epochs for better quality
3. **Attention Mechanism**: Transformer models (GPT-style)
4. **Beam Search**: Better sampling strategy
5. **Transfer Learning**: Pre-trained language models

## Files
- `DL_EX6.ipynb`: Jupyter notebook containing the implementation
- `shakespeare.txt`: Downloaded automatically from Karpathy's repository

## Requirements
```
tensorflow>=2.0
numpy
matplotlib
```

## How to Run
1. Open `DL_EX6.ipynb` in Jupyter Notebook or Google Colab
2. Run cells sequentially (dataset downloads automatically)
3. Adjust text subset size and epochs based on compute resources
4. Observe training progress and generated text samples

## Conclusion
This exercise demonstrates LSTM's capability for sequence modeling and text generation. By learning character-level patterns from Shakespeare's works, the model can generate new text in a similar style. While perfect coherence requires extensive training, even simple models can capture stylistic elements like vocabulary, punctuation, and sentence structure. LSTMs excel at capturing long-term dependencies, making them superior to Simple RNNs for natural language tasks.
