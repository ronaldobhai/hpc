# Assignment 3: QA - Convolutional Neural Networks (Fashion MNIST)

---

## Conceptual CNN Questions

### Q1: What is a convolutional layer and how does it work?
**A:** A Conv2D layer applies multiple small 3×3 (or other size) filters across an image:
- Each filter slides over the image (like a window)
- Calculates weighted sum (dot product) at each position
- Creates feature maps showing where patterns were detected
- Much more parameter-efficient than fully connected for images

### Q2: What is a "kernel" or "filter" in CNN?
**A:** 
- Small matrix (e.g., 3×3) of learned weights
- Detects specific patterns in images (edges, corners, textures)
- Multiple filters learn different patterns
- Each filter applied to entire image (weight sharing)

### Q3: How does weight sharing reduce parameters?
**A:**
- Single 3×3 filter: 9 parameters
- Applied to entire 28×28 image: same 9 parameters detect pattern anywhere
- Dense equivalent: separate weights for each location = massive parameters
- Same pattern likely appears multiple places, so sharing makes sense

### Q4: What does MaxPooling do and why?
**A:**
- Divides image into 2×2 (or other size) regions
- Keeps maximum value from each region
- Reduces spatial dimensions: 26×26 → 13×13
- Provides translation invariance: small shifts don't change output

### Q5: Why is translation invariance important?
**A:**
- Real images vary (object might be slightly left or right)
- MaxPooling: A 2×2 region with [1, 1, 2, 1] values outputs 2
- If object shifts by 1 pixel, might output [1, 2, 1, 1] = still 2
- Helps model recognize objects despite small position variations

### Q6: What is the difference between Conv2D and Dense layers?
**A:**
| Aspect | Conv2D | Dense |
|--------|--------|-------|
| **Connectivity** | Local (3×3 region) | Global (all inputs) |
| **Parameters** | Shared across space | Unique per connection |
| **Input Structure** | 2D images | 1D vectors |
| **Feature Type** | Spatial patterns | Any relationships |
| **Efficiency** | Few parameters | Many parameters |
| **Interpretability** | Detects visual features | Learned abstractions |

### Q7: Why reshape data to (28, 28, 1) instead of flattening to (784,)?
**A:**
- (28, 28, 1) preserves spatial structure: nearby pixels related
- (784,) treats all pixels equally: loses spatial information
- CNN needs 4D shape: (batch, height, width, channels)
- Dense networks need 2D: (batch, features)

### Q8: What is a feature map?
**A:**
- Output of applying a single filter to the input image
- If Conv2D(32, 3×3), produces 32 feature maps (one per filter)
- Each feature map highlights where that filter's pattern appears
- Size: If input 28×28, output ~26×26 (depending on padding)

### Q9: Why use ReLU activation in Conv layers?
**A:**
- Conv layer is linear (just matrix multiplication)
- ReLU adds non-linearity: f(x) = max(0, x)
- Allows learning non-linear relationships
- Without ReLU: convolution just becomes another linear transformation

### Q10: What would happen without MaxPooling layers?
**A:**
- No spatial downsampling
- Dimensions stay same: 28×28 → 26×26 → 24×24 (decreases by filter size)
- More parameters to learn in later layers
- Slower computation
- Might be harder to learn (too much detail, not enough abstraction)

---

## Implementation Questions

### Q11: Why is input_shape=(28, 28, 1) needed for first layer but not later?
**A:**
- First layer needs to know input dimensions
- Later layers infer input from previous layer output
- Input shape: (batch_size, 28, 28, 1) where batch_size is automatic
- Specified as (28, 28, 1) without batch dimension

### Q12: How do you choose kernel size (3×3 vs 5×5 vs 7×7)?
**A:**
- **3×3**: Detects fine details (edges), requires more layers
- **5×5**: Medium patterns, fewer layers needed
- **7×7**: Large patterns, might miss fine details
- **3×3 recommended**: Most common, balances detail and efficiency

### Q13: How do you choose number of filters (32 vs 64 vs 128)?
**A:**
- **More filters**: Better feature extraction but more parameters
- **32, 64, 128**: Progressive increase captures hierarchical features
- **Dataset size**: Smaller datasets need fewer filters (avoid overfitting)
- **Image complexity**: Simple images (MNIST) need fewer, complex need more

### Q14: Why would you use Dropout between Conv and Dense layers?
**A:**
- Conv layers extract features
- Dense layers learn classification from features
- Dropout prevents overfitting in dense layers specifically
- 0.25 (25%) aggressive after features, 0.2 (20%) lighter after second conv

### Q15: What if you have 3-channel (RGB) images instead of 1 (grayscale)?
**A:**
- Change input_shape=(28, 28, 3)
- First Conv2D automatically adjusts: filters become 3×3×3 (not 3×3×1)
- Rest of architecture stays same
- RGB contains more information, might improve accuracy

### Q16: How would you stack more Conv layers?
**A:**
```python
Conv2D(32, 3×3, relu),
MaxPool,
Conv2D(64, 3×3, relu),
MaxPool,
Conv2D(128, 3×3, relu),  # Additional layer
MaxPool,
Flatten,
Dense(...)
```
- More Conv layers → more feature learning
- Risk of overfitting with small dataset
- Diminishing returns (deeper doesn't always mean better)

### Q17: What does Flatten() do exactly?
**A:**
- Converts 5×5×64 = 1,600 values to vector of shape (1600,)
- Enables transition from Conv (2D-aware) to Dense (vector-based)
- No learnable parameters
- Necessary bridge between feature extraction and classification

### Q18: Why evaluate on test set during training?
**A:**
- Monitors generalization (not just training performance)
- Detects overfitting: if test accuracy plateaus while training improves
- Helps decide when to stop training (early stopping)
- Validation_data parameter used automatically

### Q19: What would batch_size=1 vs batch_size=128 mean?
**A:**
- **batch_size=1**: Update weights after each image (noisy, unstable)
- **batch_size=128**: Average gradients over 128 images (stable, efficient)
- **Memory**: Larger batches use more RAM
- **Speed**: Larger batches fewer updates per epoch, but GPU parallelism better
- **Convergence**: Sweet spot usually 32-128

### Q20: How would you use model.summary() to understand architecture?
**A:**
```python
model.summary()
# Shows each layer type, output shape, number of parameters
# Example:
# Conv2D(32, 3×3) from (28,28,1) → (26,26,32) with 320 parameters
# MaxPool(2×2) from (26,26,32) → (13,13,32) with 0 parameters
```
- Helps verify model structure
- Identifies layers with most parameters
- Catches shape mismatches early

---

## Architecture Design Questions

### Q21: Why two Conv layers instead of one large Conv layer?
**A:**
- **Two 3×3 Conv layers**: Receptive field = 5×5 (see larger area)
- **Parameters**: 2 × (32×3×3) << 1 × (128×5×5)
- **Non-linearity**: ReLU between layers adds more non-linearity
- **Hierarchical**: First layer detects simple patterns, second combines them

### Q22: Why progressively increase filters (32 → 64 → 128)?
**A:**
- First layer: Detects basic patterns (needs fewer filters)
- Second layer: Detects combinations of patterns (needs more)
- Third layer (if added): Even more complex patterns
- Progressive increase mirrors increasing abstraction

### Q23: Could you use just MaxPooling without Conv?
**A:** No, because:
- MaxPooling alone doesn't learn anything (just downsampling)
- Discards information without extracting features
- Would need Conv to detect patterns
- MaxPool just reduces data for efficiency

### Q24: What's the purpose of having different Dropout rates (0.25 vs 0.2)?
**A:**
- First Dropout (0.25): After first Conv layer, more aggressive
- Second Dropout (0.2): After second Conv layer, less aggressive
- Dense Dropout (0.25): Before final layer
- Fine-tuning regularization at different points
- Prevents overfitting at different levels

### Q25: Why Flatten before Dense instead of using GlobalAveragePooling?
**A:**
- **Flatten**: Preserves all spatial information from Conv layers
- **GlobalAveragePooling**: Averages each 5×5×64 down to single value per channel
- **Trade-off**: Flatten more features, GlobalPool less parameters
- Flatten chosen for better accuracy on this dataset

---

## Mathematical Questions

### Q26: What is the output size after Conv2D(32, 3×3) on 28×28 input?
**A:** 
- Formula: output_size = input_size - kernel_size + 1
- 28 - 3 + 1 = 26
- Output: 26×26×32 (26 height, 26 width, 32 filters)

### Q27: What is output size after MaxPool(2×2) on 26×26 input?
**A:**
- Formula: output_size = input_size / pool_size
- 26 / 2 = 13
- Output: 13×13×32

### Q28: How many parameters in Conv2D(32, 3×3, input_channels=1)?
**A:**
- Weights: 32 × 3 × 3 × 1 = 288
- Bias: 32 (one per filter)
- Total: 320 parameters
- Much less than Dense(input, 32) which would be millions

### Q29: How many parameters in Conv2D(64, 3×3, input_channels=32)?
**A:**
- Weights: 64 × 3 × 3 × 32 = 18,432
- Bias: 64
- Total: 18,496 parameters
- Shared weights: Same 3×3 filters applied everywhere (not per location)

### Q30: How many parameters in Flatten().output?
**A:**
- No parameters to learn
- Just reshapes: 5×5×64 = 1,600 values → vector of 1,600
- Computational: Matrix multiplication not done here

---

## Comparison Questions

### Q31: Why is CNN better than Dense for images?
**A:**
- **Spatial structure**: CNN respects 2D structure, Dense ignores it
- **Parameters**: CNN much fewer (shared weights)
- **Interpretability**: Conv filters show visual patterns
- **Translation invariance**: MaxPool handles position changes
- **Hierarchy**: Progressive feature learning
- **Performance**: CNN ~94% vs Dense ~92% on Fashion MNIST

### Q32: Would CNN work well for text? Why or why not?
**A:**
- **Not ideal** because:
  - Word order matters (spatial structure different than images)
  - RNN/LSTM better captures sequences
  - Conv filters would lose sequential dependencies
- **Could work** for:
  - Character-level analysis (local patterns matter)
  - Sentiment detection (some phrases are local patterns)
- **Better choice**: RNN for sequential text

### Q33: How would you apply CNN to 3D data (e.g., video)?
**A:**
- Use Conv3D instead of Conv2D
- Kernels: 3×3×3 (height, width, time)
- Captures spatial AND temporal patterns
- Memory intensive but can learn motion patterns

### Q34: Can you use same architecture for 224×224 images?
**A:** Yes, but:
- Same architecture works (architecture independent of input size)
- More computation (larger feature maps throughout)
- Might need more layers for larger images
- More data might be needed
- Training takes longer

### Q35: How would pretrained models (ImageNet) help?
**A:**
- ImageNet: Pre-trained on 1M+ diverse images
- Transfer learning: Copy Conv layers (learned patterns apply broadly)
- Fine-tune last layers: Only learn class-specific differences
- Benefits: Better accuracy with less training data

---

## Debugging Questions

### Q36: What if training accuracy is 50% (random guessing)?
**A:**
- Model not learning anything
- Check: Is data normalized (0-1)?
- Check: Are labels correct (0-9)?
- Check: Is input shape correct (28, 28, 1)?
- Try: Different optimizer, higher learning rate, simpler model

### Q37: What if test accuracy stays same but training improves?
**A:**
- Overfitting: Model memorizes training data
- Solutions:
  - More Dropout (0.3 or 0.4 instead of 0.25)
  - Fewer filters (32 → 16)
  - Fewer Dense neurons (128 → 64)
  - More data or augmentation
  - Early stopping

### Q38: What if model crashes with "shape mismatch" error?
**A:**
- Check input_shape matches data (28, 28, 1)
- Verify data reshaped correctly: (60000, 28, 28, 1)
- Print x_train.shape to verify
- Print layer output shapes: model.summary()

### Q39: What if Conv2D output shapes keep shrinking?
**A:**
- Each Conv2D reduces size: 28→26→24→22...
- Use "same" padding: `Conv2D(32, 3×3, padding='same')`
- Maintains size: 28→28→28
- Loses information at edges without padding

### Q40: How to interpret high loss but high accuracy?
**A:**
- Possible with softmax: Loss measures probability, not just right/wrong
- Example: True class probability = 0.99, loss = -ln(0.99) ≈ 0.01 (high for regression, low here)
- Accuracy: Right class? Yes or No (0% or 100%)
- Loss is more nuanced than accuracy

---

## Optimization Questions

### Q41: How would you increase model accuracy from 92% to 96%?
**A:**
1. More Conv layers: Learn more features
2. More filters: 32→64, 64→128
3. Larger Dense layers: 128→256
4. Data augmentation: Rotate, shift, zoom images
5. Longer training: More epochs
6. Regularization tuning: Adjust Dropout
7. Better preprocessing: Contrast normalization
8. Ensemble: Average predictions of multiple models

### Q42: How would you deploy this model to mobile?
**A:**
1. Convert to TFLite: `tf.lite.TFLiteConverter.from_keras_model(model)`
2. Quantization: Reduce precision (float32→int8)
3. Pruning: Remove non-important weights
4. Model compression: Smaller file size
5. Load on mobile: Android/iOS TFLite libraries
6. Inference: Run predictions on device

### Q43: How would you train on larger images (ImageNet size)?
**A:**
- Input: 224×224 instead of 28×28
- More Conv layers: Simple architecture insufficient
- More filters: 64→128→256→512
- Batch normalization: Stabilizes training with deep models
- Data augmentation: Critical with more parameters
- Learning rate scheduling: Adjust during training
- GPU/TPU: Much slower without GPU

### Q44: How to handle imbalanced classes (e.g., 90% one class)?
**A:**
1. **Class weights**: penalize misclassifying rare class
2. **Oversampling**: Duplicate rare class samples
3. **Undersampling**: Reduce majority class
4. **Different metrics**: F1-score, precision-recall
5. **Data augmentation**: Specifically for rare class
6. **Ensemble**: Different models trained on different splits

### Q45: How would you add real-time data augmentation?
**A:**
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2
)

model.fit(datagen.flow(x_train, y_train, batch_size=32), epochs=10)
```
- Generates new variations on-the-fly
- Prevents overfitting
- Simulates real-world variations

---

## Advanced Questions

### Q46: What is batch normalization and when to use it?
**A:**
- Normalizes activations: mean=0, std=1 for each layer
- Speeds convergence, reduces internal covariate shift
- Usually after Conv before activation: `Conv2D() → BatchNorm → ReLU`
- Good for deep networks (>6 layers)
- Current model (shallow) doesn't need it

### Q47: What is residual learning (ResNet)?
**A:**
- Skip connections: Output of layer added to input
- Solves vanishing gradient in very deep networks
- Enables training of 100+ layer networks
- Formula: `y = Conv(x) + x` (skip connection)
- Not needed for shallow networks

### Q48: How would you visualize what Conv filters learned?
**A:**
```python
# Get filter weights
filters = model.layers[0].get_weights()[0]  # Shape: (3, 3, 1, 32)

# Visualize some filters
for i in range(32):
    plt.subplot(4, 8, i+1)
    plt.imshow(filters[:, :, 0, i], cmap='gray')
    plt.axis('off')
plt.show()
```
- First layer filters show edge/corner detectors
- Visualizes what network learns

### Q49: How to use saliency maps (which pixels matter)?
**A:**
```python
# Compute gradient of output w.r.t. input
gradient = compute_gradients(model, image)

# Visualize: Bright = important pixels
plt.imshow(gradient, cmap='hot')
```
- Shows which pixels influence prediction
- Useful for interpretability
- Debugging: Why does model predict class X?

### Q50: What is transfer learning and why is it powerful?
**A:**
- Use pre-trained model (e.g., ImageNet weights)
- Remove final layers, add new layers for your task
- Train only new layers (transfer knowledge)
- Advantages:
  - Better accuracy with less data
  - Faster training
  - Learned features transfer broadly
- Example: Classify cars using ImageNet pre-trained weights

---

## Summary Key Points

1. **Convolution detects patterns** through learned filters
2. **Weight sharing** drastically reduces parameters
3. **MaxPooling** reduces dimensions and adds robustness
4. **Spatial structure** matters for images (CNN preserves it)
5. **Hierarchy** emerges: simple patterns → complex features → class
6. **ReLU activations** enable non-linear learning in Conv layers
7. **Dropout regularization** prevents overfitting
8. **Architecture design** balances capacity and overfitting
9. **Data preprocessing** (normalize, reshape) critical
10. **Generalization** (test accuracy) is ultimate goal
