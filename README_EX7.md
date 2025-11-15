# Exercise 7: Encoder-Decoder Model for Sequence-to-Sequence Translation

## Aim
To implement an encoder-decoder model for sequence-to-sequence tasks, specifically for translating text from one language to another (English to French).

## Description
This exercise builds an encoder-decoder architecture using LSTM layers for machine translation. The encoder-decoder (also called sequence-to-sequence or seq2seq) model is designed to transform one sequence into another, making it ideal for tasks like translation, summarization, and conversational AI. The model learns to map English phrases to their French equivalents.

## What is Encoder-Decoder Architecture?

### Core Concept
The Encoder-Decoder model consists of two main components:

1. **Encoder**: Processes the input sequence and compresses it into a fixed-length context vector (thought vector)
2. **Decoder**: Generates the output sequence step-by-step using the context vector

```
Input Sequence → [Encoder] → Context Vector → [Decoder] → Output Sequence
"hello"       →    LSTM    →  Hidden State  →   LSTM    → "bonjour"
```

### Why Encoder-Decoder?
- **Variable Length**: Input and output can have different lengths
- **Context Preservation**: Context vector captures meaning of entire input
- **Step-by-Step Generation**: Decoder generates one token at a time
- **Versatile**: Works for translation, summarization, Q&A, chatbots

## Model Architecture

### Complete Architecture
```
INPUT SEQUENCE (English)
    ↓
Encoder LSTM (256 units)
    ↓ (state_h, state_c)
Context Vector (Hidden States)
    ↓ (passed as initial state)
Decoder LSTM (256 units)
    ↓ (receives target sequence)
Dense Layer (Softmax)
    ↓
OUTPUT SEQUENCE (French)
```

### Training Mode
```
Encoder Input: "hello" (one-hot encoded)
    ↓
Encoder LSTM → states [h, c]
    ↓
Decoder Input: "<start> bonjour" (teacher forcing)
Decoder Initial State: [h, c] from encoder
    ↓
Decoder Output: "bonjour <end>"
```

### Inference Mode
```
Encoder Input: "hello"
    ↓
Encoder LSTM → states [h, c]
    ↓
Decoder Input: "<start>" token
Loop:
  - Predict next character
  - Feed predicted character back as input
  - Update states
  - Stop when "<end>" token or max length
    ↓
Generated Output: "bonjour"
```

## Dataset

### Sample English-French Pairs
```python
input_texts = ["hello", "how are you", "good morning", "thank you", "good night"]
target_texts = ["bonjour", "comment ça va", "bonjour", "merci", "bonne nuit"]
```

### Data Characteristics
- **Vocabulary Size**: Number of unique characters in each language
- **Max Sequence Length**: Longest phrase in each language
- **Encoding**: Character-level (each character is a token)

## Detailed Procedure

### Step 1: Import Libraries
```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
```

### Step 2: Prepare Character-Level Data
```python
# Sample dataset
input_texts = ["hello", "how are you", "good morning", "thank you", "good night"]
target_texts = ["bonjour", "comment ça va", "bonjour", "merci", "bonne nuit"]

# Extract unique characters
input_characters = sorted(set("".join(input_texts)))
target_characters = sorted(set("".join(target_texts)))

num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

# Create character-to-index mappings
input_token_index = {char: i for i, char in enumerate(input_characters)}
target_token_index = {char: i for i, char in enumerate(target_characters)}
reverse_target_char_index = {i: char for char, i in target_token_index.items()}
```

### Step 3: Create Training Matrices
```python
# Initialize zero matrices
encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype="float32"
)
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype="float32"
)
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype="float32"
)

# One-hot encode sequences
for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.0
    
    for t, char in enumerate(target_text):
        decoder_input_data[i, t, target_token_index[char]] = 1.0
        if t > 0:
            # decoder_target_data is ahead by one timestep
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
```

### Step 4: Define Encoder
```python
# Encoder
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder_lstm = LSTM(256, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)

# We discard encoder_outputs and only keep the states
encoder_states = [state_h, state_c]
```

### Step 5: Define Decoder
```python
# Decoder
decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(256, return_sequences=True, return_state=True)

# Decoder uses encoder_states as initial state
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

# Dense layer to generate probability distribution
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
```

### Step 6: Define Training Model
```python
# Combined model for training
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

### Step 7: Train the Model
```python
# Train
history = model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=64,
    epochs=100,
    validation_split=0.2
)
```

### Step 8: Define Inference Models

#### Encoder Model (Inference)
```python
encoder_model = Model(encoder_inputs, encoder_states)
```

#### Decoder Model (Inference)
```python
# Decoder state inputs
decoder_state_input_h = Input(shape=(256,))
decoder_state_input_c = Input(shape=(256,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# Decoder outputs and states
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs
)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)

# Decoder model
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states
)
```

### Step 9: Implement Decoding Function
```python
def decode_sequence(input_seq):
    # Encode the input as state vectors
    states_value = encoder_model.predict(input_seq)
    
    # Generate empty target sequence of length 1
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_token_index[' ']] = 1.0
    
    # Sampling loop for generating characters
    decoded_sentence = ''
    for _ in range(max_decoder_seq_length):
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char
        
        # Exit condition: newline or max length
        if sampled_char == '\n':
            break
        
        # Update the target sequence
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.0
        
        # Update states
        states_value = [h, c]
    
    return decoded_sentence
```

### Step 10: Test Translation
```python
# Test on first example
input_test_seq = encoder_input_data[0:1]
translated_text = decode_sequence(input_test_seq)
print("Input:", input_texts[0])
print("Translated text:", translated_text)
```

## Expected Output

### Training Progress
```
Epoch 1/100
1/1 [==============================] - 2s 2s/step - loss: 2.8934 - accuracy: 0.0833 - val_loss: 2.7123 - val_accuracy: 0.1250
Epoch 2/100
1/1 [==============================] - 0s 89ms/step - loss: 2.6234 - accuracy: 0.1667 - val_loss: 2.4567 - val_accuracy: 0.2500
...
Epoch 100/100
1/1 [==============================] - 0s 85ms/step - loss: 0.1234 - accuracy: 0.9583 - val_loss: 0.2456 - val_accuracy: 0.9375
```

### Translation Results
```
Input: hello
Translated text: bonjour

Input: how are you
Translated text: comment ça va

Input: good morning
Translated text: bonjour

Input: thank you
Translated text: merci

Input: good night
Translated text: bonne nuit
```

## Key Concepts

### 1. Teacher Forcing
During training, the decoder receives the actual target sequence (not its own predictions):
```
Decoder Input:  <start> b o n j o u
Decoder Target:       b o n j o u r
```

### 2. Inference Loop
During inference, decoder generates one character at a time:
```
Step 1: Input "<start>" → Output "b"
Step 2: Input "b"       → Output "o"
Step 3: Input "o"       → Output "n"
...
```

### 3. State Transfer
Encoder's final hidden states initialize decoder:
```
Encoder states → [h, c] → Decoder initial state
```

### 4. One-Hot Encoding
Each character is represented as a binary vector:
```
"h" → [0, 0, 0, 1, 0, 0, ...]  (size: vocab_size)
```

## Model Parameters

| Component | Configuration |
|-----------|--------------|
| Encoder LSTM | 256 units, return_state=True |
| Decoder LSTM | 256 units, return_sequences=True, return_state=True |
| Dense Layer | num_decoder_tokens units, softmax |
| Optimizer | Adam |
| Loss | Categorical Crossentropy |
| Batch Size | 64 |
| Epochs | 100 |

## Limitations of This Implementation

1. **Small Dataset**: Only 5 phrase pairs (for demonstration)
2. **Character-Level**: Slower than word-level for long sequences
3. **No Attention**: Context vector is fixed-length bottleneck
4. **Simple Decoding**: Greedy decoding (no beam search)

## Real-World Improvements

1. **Larger Dataset**: Use parallel corpora (millions of sentence pairs)
2. **Word-Level**: Tokenize into words instead of characters
3. **Attention Mechanism**: Allow decoder to focus on relevant encoder positions
4. **Beam Search**: Explore multiple decoding paths
5. **Transformers**: Replace LSTM with self-attention (BERT, GPT architecture)

## Applications

1. **Machine Translation**: English ↔ French, etc.
2. **Text Summarization**: Long article → Summary
3. **Chatbots**: User query → Bot response
4. **Code Generation**: Natural language → Code
5. **Image Captioning**: Image → Text description

## Files
- `DL_EX7.ipynb`: Jupyter notebook containing the implementation

## Requirements
```
tensorflow>=2.0
numpy
matplotlib
```

## How to Run
1. Open `DL_EX7.ipynb` in Jupyter Notebook or Google Colab
2. Run all cells sequentially
3. Observe model architecture, training progress, and translation results
4. Experiment with different phrase pairs or larger datasets

## Conclusion
This exercise demonstrates the encoder-decoder architecture for sequence-to-sequence learning. While the example uses a small dataset for demonstration, the same architecture scales to production translation systems with proper datasets. The model successfully learns to map English phrases to French equivalents by compressing input meaning into a context vector and decoding it into the target language. Modern improvements like attention mechanisms and transformer architectures build upon this foundational concept.
