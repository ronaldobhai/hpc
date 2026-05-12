# Assignment 3: Convolutional Neural Networks - Fashion MNIST Classification

## Overview
Fashion MNIST is a dataset of 70,000 grayscale 28×28 pixel images of 10 fashion items (T-shirts, shoes, bags, etc.). This assignment uses Convolutional Neural Networks (CNNs) to classify images with higher accuracy than dense networks.

---

## Dataset Loading

### Data Import
```python
from tensorflow.keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
```

### Fashion Classes
```
0: T-shirt/top
1: Trouser
2: Pullover
3: Dress
4: Coat
5: Sandal
6: Shirt
7: Sneaker
8: Bag
9: Ankle boot
```
- 10-class classification problem
- Similar to MNIST but clothing items instead of digits
- Dataset balanced: ~6,000 samples per class

---

## Data Preprocessing

### Type Conversion & Normalization
```python
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
```
- Converts from uint8 to float32 (enables division)
- Normalizes pixel values from [0, 255] to [0, 1]
- Neural networks train better with normalized data

### Reshaping for CNN
```python
x_train = x_train.reshape(-1, 28, 28, 1)  # (60000, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)   # (10000, 28, 28, 1)
```

**Reshape Explanation:**
- `-1`: Automatically infers number of samples (60,000)
- `28, 28`: Height and width of images
- `1`: Number of channels (1 = grayscale, 3 = RGB)
- Format: (batch_size, height, width, channels) = NHWC format

**Why 4D instead of 2D?**
- Dense networks: Flatten to (60000, 784) - loses spatial structure
- CNN: Keep 4D shape - preserves spatial relationships between pixels
- Nearby pixels are more likely to contain related features

---

## CNN Model Architecture

### Full Model
```python
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.2),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.25),
    Dense(10, activation='softmax')
])
```

### Layer Explanation

#### Layer 1: Conv2D (32 filters, 3×3 kernel)
```python
Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1))
```

**What Conv2D does:**
- Applies 32 small 3×3 matrices (filters) over entire image
- Each filter detects specific patterns (edges, corners, textures)
- Slides filter across image: calculates dot product at each position
- Creates feature maps (output of 32 filters)

**Convolution Process:**
```
Original pixel values:        Filter (3×3):          Result:
[1 2 3]                      [0.1 0.2 0.1]          0.1×1 + 0.2×2 + ...
[4 5 6] × [0.2 0.5 0.2]  →  = weighted sum
[7 8 9]   [0.1 0.2 0.1]     (single output value)
```

**Output shape:** 28×28 input → 26×26 output (without padding)
- 3×3 filter: 28 - 3 + 1 = 26
- 32 filters: 26×26×32 feature maps

**Kernel Size (3, 3):**
- Detects local patterns in 3×3 regions
- Preserves edge information better than larger kernels
- Common choice (alternatives: 5×5, 7×7)

**ReLU Activation:**
- Adds non-linearity within convolution
- Allows learning complex patterns

#### Layer 2: MaxPooling2D (2×2)
```python
MaxPooling2D((2, 2))
```

**What MaxPooling does:**
- Divides image into 2×2 regions
- Keeps maximum value from each region
- Discards 3 out of 4 values per region

**MaxPooling Example:**
```
Input 4×4:            2×2 pools:        Max values:
[1 3 2 4]            [1 3] [2 4]        [3  4]
[5 2 1 2]  →  max of [5 2] [1 2]   →   [5  2]
[4 1 3 2]            [4 1] [3 2]
[1 2 4 3]            [1 2] [4 3]
```

**Benefits:**
1. **Dimension reduction**: 26×26 → 13×13 (1/4 the size)
2. **Spatial invariance**: Slight shifts don't change max value
3. **Parameter reduction**: Less computation downstream
4. **Noise reduction**: Only important (maximum) values preserved

#### Layer 3: Dropout
```python
Dropout(0.25)  # Drop 25% of neurons
```
- Prevents overfitting by deactivating random neurons during training
- Encourages robustness (network can't rely on specific neurons)
- Used after feature extraction for regularization

#### Layer 4: Conv2D (64 filters, 3×3 kernel)
```python
Conv2D(64, (3, 3), activation='relu')
```
- Second convolution layer with 64 filters (more than first layer)
- Detects higher-level patterns using first layer's features as input
- **Input shape**: 13×13×32 (from previous MaxPool)
- **Output shape**: 11×11×64 (from convolution)

**Why more filters?**
- First layer: Detects basic features (edges, corners)
- Second layer: Detects compound features (shapes, objects)
- More filters needed for complex patterns

#### Layer 5: MaxPooling2D (2×2)
```python
MaxPooling2D((2, 2))
```
- Further reduces spatial dimensions: 11×11 → 5×5
- Cumulative downsampling: 28×28 → 26×26 → 13×13 → 11×11 → 5×5
- Reduces parameters significantly

#### Layer 6: Dropout
```python
Dropout(0.2)  # Drop 20% of neurons
```
- Slightly less aggressive than first dropout (20% vs 25%)

#### Layer 7: Flatten
```python
Flatten()
```
- Converts 5×5×64 = 1,600 values to 1D vector
- Bridges CNN feature extraction with dense classification layers
- No parameters, just reshapes

#### Layer 8: Dense (128 neurons)
```python
Dense(128, activation='relu')
```
- Takes 1,600 features, learns 128 abstract representations
- ReLU activation for non-linearity
- Learns which feature combinations indicate which class

#### Layer 9: Dropout
```python
Dropout(0.25)
```
- Regularization before final classification layer

#### Layer 10: Dense (10 neurons, Softmax)
```python
Dense(10, activation='softmax')
```
- Output layer: 10 neurons for 10 classes
- Softmax: Converts to probability distribution (sum = 1)
- One neuron per class (T-shirt, Trouser, etc.)

---

## Compilation

```python
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

### Loss Function: Sparse Categorical Cross-Entropy
- **"Sparse"**: Labels are integers (0-9), not one-hot encoded
- Formula: `-log(p[true_class])`
- Example: True label = 3, prediction = [0.01, 0.05, 0.1, 0.8, ...] → loss = -log(0.8) ≈ 0.22

**Vs Categorical Cross-Entropy:**
- Categorical: Requires one-hot encoded labels [0, 0, 0, 1, 0, ...]
- Sparse: Accepts integer labels (3)
- Same mathematically, different input format

---

## Training

```python
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
```

### Training Process
- 60,000 samples processed in batches (default batch_size=32)
- 10 epochs = 10 complete passes through data
- Validation on test set each epoch monitors generalization

### Expected Performance
- Training: ~95-98% accuracy
- Test: ~92-94% accuracy
- Slight gap normal (some overfitting)

---

## Evaluation

```python
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test Accuracy:", test_acc)
print("Test Loss:", test_loss)
```

### Key Metrics
- **Test Accuracy**: Percentage of correctly classified fashion items
- **Test Loss**: Sparse categorical cross-entropy on test set
- Both computed on 10,000 unseen test images

---

## Why CNNs are Better for Images

### Parameter Count Comparison
```
Dense only:
784 → 128 neurons = 784 × 128 = 100,352 parameters

CNN:
Conv2D(32, 3×3) = 32 × 3 × 3 × 1 = 288 parameters (shared across image)
Conv2D(64, 3×3) = 64 × 3 × 3 × 32 = 18,432 parameters
Total much less despite better performance
```

### Why Shared Weights Work
- Pattern (edge, corner) likely appears multiple locations in image
- Same filter (weights) can detect pattern anywhere
- No need to learn separate weights for each position
- **Weight sharing**: Single 3×3 filter applied to entire image

### Why Local Connectivity Helps
- Nearby pixels related (form edges, shapes)
- CNN respects this structure
- Dense network treats all pixels equally (ignores structure)
- CNN learns hierarchical features

### Feature Extraction
- **Layer 1**: Detects edges, basic shapes
- **Layer 2**: Detects combinations of edges (shoes, collars, etc.)
- **Dense layers**: Learn which combinations indicate which class
- Progressive abstraction from pixels to class

---

## Architecture Design Decisions

### Why 32 then 64 filters?
- Progressive increase in feature complexity
- 32 sufficient for basic features
- 64 captures more intricate patterns
- Could use 128, 256 for complex datasets

### Why 3×3 kernel size?
- Smallest receptive field for edges
- Good balance between capture and parameters
- Very common in practice
- Alternatives: 5×5 (larger), 1×1 (pointwise)

### Why MaxPooling after Conv?
- Reduces spatial dimensions → less computation
- Adds translation invariance
- Prevents overfitting to exact pixel positions
- Alternative: Strides in convolution (but pooling more common)

### Why Dropout?
- CNNs have many parameters (can overfit)
- Dropout random deactivation prevents co-adaptation
- Especially important with smaller datasets

---

## Summary of Key Concepts

1. **Convolution**: Sliding window pattern detection
2. **Filters**: Small learned matrices detecting patterns
3. **Feature Maps**: Output of applying filter to input
4. **MaxPooling**: Spatial downsampling with information preservation
5. **Hierarchical Learning**: Progressive abstraction from pixels to classes
6. **Weight Sharing**: Same filter detects patterns everywhere
7. **Local Connectivity**: Respects spatial structure of images
8. **Translation Invariance**: MaxPooling handles small shifts
9. **Parameter Efficiency**: Far fewer parameters than Dense networks
10. **Generalization**: Better test performance than Dense with same data
