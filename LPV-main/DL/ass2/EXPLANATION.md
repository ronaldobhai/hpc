# Assignment 2: Deep Learning with Convolutional and Dense Networks

## Part 1: MNIST Digit Recognition (Optical Character Recognition)

### Overview
MNIST (Modified National Institute of Standards and Technology) contains 70,000 handwritten digit images (0-9). This assignment builds a neural network to recognize digits with high accuracy.

---

## Part 1: MNIST OCR

### Dataset Loading
```python
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```
- **x_train**: 60,000 training images, each 28×28 pixels, grayscale (0-255)
- **y_train**: 60,000 labels (0-9, digit class)
- **x_test**: 10,000 test images, 28×28 pixels
- **y_test**: 10,000 test labels

### Image Visualization
```python
plt.imshow(x_train[0], cmap='gray')  # Display first training image
plt.show()                           # Shows grayscale handwritten digit
```
- `cmap='gray'`: Displays pixel values as grayscale (0=black, 255=white)
- Each image is a 2D array of pixel intensities

### Reshaping Data
```python
x_train = x_train.reshape(60000, 784)  # 28×28 = 784 features
x_test = x_test.reshape(10000, 784)
```
- Converts 2D images (28, 28) to 1D vector (784,)
- Each pixel becomes one feature for the neural network
- 784 features = 60,000 training samples (matrix shape: 60000×784)

### Data Type Conversion
```python
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
```
- Converts from uint8 (0-255) to float32
- Allows mathematical operations and prevents overflow

### Feature Normalization
```python
x_train /= 255  # Pixel values now in range [0, 1]
x_test /= 255
```
- Divides by max pixel value (255)
- Normalizes to [0, 1] range
- Neural networks train better with normalized inputs
- Equivalent to: `x_train = x_train / 255`

### One-Hot Encoding
```python
num_classes = 10
y_train = np.eye(num_classes)[y_train]  # Convert labels to one-hot
y_test = np.eye(num_classes)[y_test]
```
- **Original label**: `3` (digit three)
- **One-hot encoded**: `[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]` (10 values, 1 at position 3)
- Enables multi-class classification with softmax output
- Each position represents probability for that digit class

**np.eye() explanation:**
- `np.eye(10)` creates 10×10 identity matrix
- `np.eye(10)[3]` returns row 3: `[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]`
- Vectorized indexing applies this to all labels at once

---

## Model Architecture

### Sequential Model
```python
model = Sequential([
    Dense(512, activation='relu', input_shape=(784,)),     # Layer 1
    Dropout(0.2),                                           # Regularization
    Dense(512, activation='relu'),                          # Layer 2
    Dropout(0.2),                                           # Regularization
    Dense(num_classes, activation='softmax')                # Output layer
])
```

**Layer-by-Layer Breakdown:**

1. **Input Dense Layer**
   - Input: 784 neurons (one per pixel)
   - Output: 512 neurons
   - Activation: ReLU (introduces non-linearity)
   - `input_shape=(784,)`: Specifies input dimension

2. **First Dropout Layer**
   - `Dropout(0.2)`: Randomly deactivates 20% of neurons during training
   - **Why?** Prevents co-adaptation, reduces overfitting
   - **How?** During training, 80% of neurons are active
   - During inference (test), all neurons active but outputs scaled by 0.8

3. **Second Dense Layer**
   - Input: 512 neurons (from previous layer)
   - Output: 512 neurons (same size, learns more patterns)
   - Activation: ReLU
   - No input_shape needed (inferred from previous layer)

4. **Second Dropout Layer**
   - `Dropout(0.2)`: Again randomly deactivates 20%
   - Multiple dropouts help regularization

5. **Output Dense Layer**
   - Input: 512 neurons
   - Output: 10 neurons (one per digit class)
   - Activation: **Softmax** (converts to probabilities)
   - Sum of 10 outputs = 1.0 (probability distribution)

### Softmax Activation
```python
activation='softmax'
```
- Formula: `σ(x_i) = e^(x_i) / Σ(e^(x_j))` for j in all classes
- Converts network outputs to probability distribution
- Example output: `[0.01, 0.05, 0.88, 0.02, ...]` (88% confident it's digit 2)
- Required for multi-class classification with categorical cross-entropy

---

## Compilation

```python
model.compile(
    loss='categorical_crossentropy',     # Loss function for multi-class
    optimizer=RMSprop(),                 # Optimization algorithm
    metrics=['accuracy']                 # Metric to monitor
)
```

### Loss: Categorical Cross-Entropy
- **Formula**: `-Σ(y_true × log(y_pred))`
- **Use case**: Multi-class classification with one-hot encoded labels
- Measures difference between predicted probabilities and true labels
- Lower is better
- Heavily penalizes wrong confident predictions

**Example:**
- True label: `[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]` (digit 3)
- Prediction 1: `[0.05, 0.05, 0.05, 0.8, 0.0, ...]` → low loss (good)
- Prediction 2: `[0.1, 0.1, 0.1, 0.1, 0.1, ...]` → high loss (bad)

### Optimizer: RMSprop
- Adaptive learning rate optimization
- Divides learning rate by exponential moving average of gradients
- Good for multi-layer networks
- Alternative: Adam (usually similar, RMSprop is classic for this task)

### Metric: Accuracy
- Percentage of correct predictions: `(# correct) / (total samples)`
- Doesn't affect training, just monitoring
- For balanced datasets (10% per class in MNIST)

---

## Training

```python
batch_size = 128
epochs = 20

history = model.fit(
    x_train,                                    # Training data
    y_train,                                    # Training labels
    batch_size=batch_size,                      # 128 samples per update
    epochs=epochs,                              # 20 full passes through data
    verbose=1,                                  # Print progress each epoch
    validation_data=(x_test, y_test)            # Evaluate on test set each epoch
)
```

**Training Process:**
1. **Epoch 1**: 60,000 / 128 ≈ 470 batches, each updates weights
2. After each epoch: Evaluate on test set (10,000 images)
3. Repeat for 20 epochs
4. Total weight updates: 20 × 470 ≈ 9,400

**Hyperparameters:**
- `batch_size=128`: Memory efficient, fewer weight updates per epoch
- `epochs=20`: ~98% accuracy typically reached, diminishing returns after
- `validation_data=(x_test, y_test)`: Monitors test accuracy each epoch

---

## Part 2: IMDB Sentiment Analysis

### Dataset Overview
```python
from tensorflow.keras.datasets import imdb
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)
```
- **IMDB**: 50,000 movie reviews (25,000 train, 25,000 test)
- **num_words=10,000**: Use only top 10,000 most common words
- **X_train**: Each review is list of integer word IDs
- **y_train**: Binary labels (0=negative, 1=positive sentiment)

**Example review structure:**
```python
X_train[0] = [1, 14, 22, 16, ...]  # Word indices, not words
y_train[0] = 1                      # Positive review
```

### Decoding Reviews
```python
index = imdb.get_word_index()                    # word → ID mapping
reverse_index = dict((value, key) for (key, value) in index.items())  # ID → word

decoded_review = " ".join([reverse_index.get(i-3, "#") for i in data[0]])
```
- `get_word_index()`: Dictionary mapping words to IDs
- IDs 0-2 are reserved (padding, start, unknown)
- `reverse_index.get(i-3, "#")`: Convert ID back to word (offset by 3)
- `#` is placeholder for unknown words

**Output example:**
```
"this film was really great and ..."  # Decoded movie review text
```

### Vectorization
```python
def vectorize(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension), dtype='float32')
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.  # One-hot encoding for each word
    return results

data = vectorize(data)
```

**Process:**
- **Input**: Review = `[1, 14, 22, 16, ...]` (word indices)
- **Output**: 10,000-dimensional vector with 1s at word positions, 0s elsewhere
- **Example**: Word ID 14 appears → `[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, ...]`
- Converts variable-length sequences to fixed-size vectors

### Analysis
```python
length = [len(i) for i in X_train]
print("Average Review Length:", np.mean(length))      # ~238 words
print("Standard Deviation    :", round(np.std(length)))  # ~178 words
```
- Reviews have variable lengths (some 10 words, some 2000+)
- Vectorization fixes this to 10,000 features
- Most words don't appear in most reviews (sparse data)

### Label Balance
```python
sns.countplot(x='label', data=labelDF)
```
- Perfect balance: 12,500 positive + 12,500 negative reviews
- No class imbalance issues (unlike many real datasets)

---

## Model Architecture for Sentiment

```python
model = Sequential([
    Dense(128, activation='relu', input_shape=(10000,)),  # Flatten: 10k features
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')                        # Binary classification
])
```

**Architecture:**
- **Input**: 10,000 features (vectorized review)
- **Hidden 1**: 128 neurons, ReLU
- **Dropout 1**: 30% dropped (more aggressive than MNIST's 20%)
- **Hidden 2**: 64 neurons, ReLU
- **Dropout 2**: 30% dropped
- **Output**: 1 neuron, Sigmoid (binary probability)

### Sigmoid Activation
```python
activation='sigmoid'
```
- Formula: `σ(x) = 1 / (1 + e^(-x))`
- Outputs value between 0 and 1
- Single neuron represents P(positive review)
- If output ≥ 0.5: predict positive (1)
- If output < 0.5: predict negative (0)

### Compilation for Sentiment
```python
model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(),
    metrics=['accuracy']
)
```

**Binary Cross-Entropy:**
- Formula: `-[y*log(y_pred) + (1-y)*log(1-y_pred)]`
- For single neuron outputting probability
- Penalizes confident wrong predictions

---

## Key Differences: MNIST vs IMDB

| Aspect | MNIST | IMDB |
|--------|-------|------|
| **Input Type** | Images (28×28 pixels) | Text (words) |
| **Features** | 784 (pixel values) | 10,000 (word IDs) |
| **Output Classes** | 10 (digits 0-9) | 2 (positive/negative) |
| **Output Activation** | Softmax | Sigmoid |
| **Loss Function** | Categorical crossentropy | Binary crossentropy |
| **Dropout** | 20% | 30% |
| **Architecture** | 2 hidden layers (512 neurons) | 2 hidden layers (128, 64) |

---

## Summary of Key Concepts

1. **Data Preprocessing**: Reshape, normalize, encode labels appropriately
2. **One-Hot Encoding**: Converts class labels to categorical format
3. **Vectorization**: Converts variable-length sequences to fixed-size
4. **Dropout**: Regularization technique preventing overfitting
5. **Softmax vs Sigmoid**: Multi-class vs binary classification
6. **Categorical vs Binary Cross-Entropy**: Different loss functions
7. **Batch Training**: Gradient updates on batches, not full dataset
8. **Validation**: Monitor test performance during training
9. **ReLU Activation**: Non-linearity in hidden layers
10. **Output Layers**: Match activation and loss to task type
