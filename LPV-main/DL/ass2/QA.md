# Assignment 2: QA - MNIST OCR and IMDB Sentiment Analysis

---

## MNIST Digit Recognition - Conceptual Questions

### Q1: What does MNIST stand for?
**A:** Modified National Institute of Standards and Technology - a dataset of 70,000 handwritten digit images used for training and testing neural networks.

### Q2: Why do we reshape images from (28, 28) to (784,)?
**A:** Neural networks with Dense layers expect 1D input vectors, not 2D matrices. We flatten the 2D image into a 1D array: 28 × 28 = 784 features.

### Q3: What is the purpose of dividing pixel values by 255?
**A:** Normalization: Converts pixel intensities from [0, 255] to [0, 1]. Neural networks train better with normalized inputs (faster convergence, more stable gradients).

### Q4: Why use one-hot encoding for labels instead of keeping them as 0-9?
**A:** 
- **One-hot** (10 values): `[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]` for digit 3
- **Categorical**: label = 3
- One-hot is required for softmax output with categorical cross-entropy loss
- Treats each class as independent binary classification

### Q5: What does np.eye(10)[y_train] do?
**A:** 
- `np.eye(10)` creates a 10×10 identity matrix
- Indexing with array uses values as row indices
- For y_train = `[3, 7, 2, ...]`, returns one-hot vectors for each label
- Vectorized one-hot encoding

### Q6: What is the difference between Dense layer with ReLU and output Softmax?
**A:**
- **ReLU in hidden layers**: Introduces non-linearity, enables learning complex patterns
- **Softmax in output**: Converts 10 raw values to probability distribution (sum to 1)

### Q7: What is Dropout and why is it important?
**A:** 
- Dropout(0.2): Randomly deactivates 20% of neurons during training
- Prevents co-adaptation (neurons learning to work together)
- Reduces overfitting by forcing robustness
- Acts as ensemble of networks

### Q8: How does Dropout prevent overfitting?
**A:**
- Creates different subnetworks each training step
- Forces individual neurons to be useful, not rely on others
- During testing, all neurons active but outputs scaled
- Similar to "noise injection" regularization

### Q9: What does RMSprop optimizer do differently from basic SGD?
**A:**
- RMSprop: Uses exponential moving average of squared gradients
- Adapts learning rate based on gradient history
- Faster convergence than vanilla SGD
- Better handling of oscillations in loss surface

### Q10: Why is accuracy a good metric for MNIST?
**A:**
- Classes are perfectly balanced (10% each)
- Accuracy = correct predictions / total predictions
- Easy to interpret (97% means 97 digits correct out of 100)
- No bias toward any class

---

## MNIST Implementation Questions

### Q11: What would happen if we didn't normalize pixel values?
**A:**
- Neural network receives values in [0, 255] range
- Large input values → large gradients → unstable training
- Slower convergence or divergence
- Network might struggle with numerical precision

### Q12: Why use categorical_crossentropy loss instead of MSE for classification?
**A:**
- MSE: Treats classification as regression, not ideal
- Cross-entropy: Measures probability distribution difference
- Penalizes wrong confident predictions more heavily
- Better for classification tasks

### Q13: What's the purpose of batch_size=128?
**A:**
- Updates weights after processing 128 samples
- 60,000 / 128 ≈ 470 batches per epoch
- Balances memory usage with gradient stability
- Too small (8): Noisy, unstable updates
- Too large (1000): Requires more memory, fewer updates

### Q14: Why monitor validation on test set during training?
**A:**
- Detects overfitting: if test accuracy plateaus while training improves
- Helps decide when to stop training (early stopping)
- Gives realistic estimate of model generalization
- Can implement early stopping to prevent overfitting

### Q15: What does reshape(60000, 784) preserve?
**A:**
- 60,000 images (first dimension preserved)
- 784 features per image (28×28)
- Order of pixels preserved (deterministic reshaping)
- Data integrity maintained

### Q16: How would you modify the model for color images (RGB)?
**A:**
- Change reshape: (28, 28, 3) for RGB instead of (28, 28) for grayscale
- Flatten to 28×28×3 = 2352 features
- Input shape: input_shape=(2352,)
- Same architecture, more features to process

### Q17: What if we used 1000 hidden neurons instead of 512?
**A:**
- Larger capacity, can learn more complex patterns
- Longer training time
- Higher overfitting risk (more parameters to learn)
- Probably unnecessary for MNIST (already near optimal)

### Q18: What's the difference between x_train.astype and dividing by 255?
**A:**
- `.astype('float32')`: Changes data type (int → float), enables division
- `/= 255`: Performs actual normalization
- Both necessary: type conversion + value scaling

### Q19: How would you handle images with different sizes?
**A:**
- Resize all to 28×28 using image processing
- Or use padding/cropping to fixed size
- Or train on variable-size batches (more complex)
- MNIST already has fixed 28×28 size

### Q20: What would verbose=0 do instead of verbose=1?
**A:**
- `verbose=0`: No output during training (silent)
- `verbose=1`: Print progress each epoch (current implementation)
- `verbose=2`: Print progress once per epoch as single line
- Useful for long training runs to reduce output

---

## IMDB Sentiment Analysis - Conceptual Questions

### Q21: Why are reviews encoded as word IDs instead of actual text?
**A:**
- IDs are integers, easier for neural networks
- Reduces memory (small integers vs long strings)
- Allows vectorization to fixed-size arrays
- Can batch process multiple reviews

### Q22: What is vectorization in the context of IMDB reviews?
**A:**
- Converts variable-length review (list of word IDs) to fixed-size vector
- Creates 10,000-dimensional vector (one position per word)
- Sets 1 at positions where words appear, 0 elsewhere
- Enables processing in neural networks

### Q23: Why use num_words=10,000 instead of all words?
**A:**
- Total vocabulary > 100,000 words
- Top 10,000 covers ~97% of reviews
- Reduces dimensionality (10k vs 100k features)
- Reduces memory and computation
- Ignores rare words (less informative anyway)

### Q24: How does imdb.get_word_index() work?
**A:**
- Returns dictionary: word → word ID mapping
- 'great' → 42, 'good' → 67, etc.
- Based on frequency (common words have lower IDs)
- Allows reverse lookup: ID → word

### Q25: What does reverse_index.get(i-3, "#") do?
**A:**
- Word IDs start at 3 (0, 1, 2 are reserved)
- i-3 converts back to original indexing
- `.get(i-3, "#")`: Returns word for ID, or "#" if unknown
- # placeholder for out-of-vocabulary words

### Q26: Why is sentiment analysis a binary classification task?
**A:**
- IMDB reviews: 1-10 star ratings
- Converted to binary: positive (≥6 stars) or negative (<6 stars)
- Simplest classification task
- Could be expanded to 5-class (1-5 stars)

### Q27: What's the advantage of analyzing review length?
**A:**
- Average 238 words: reasonable length to process
- Variance (178 words): reviews have different lengths
- Helps decide vectorization strategy
- Longer reviews might need different models

### Q28: Why is label balance (12.5k positive, 12.5k negative) important?
**A:**
- Prevents class imbalance bias
- Accuracy is reliable metric
- No need for class weights or resampling
- Each class equally represented

### Q29: How would you handle variable-length reviews differently?
**A:**
- Use padding: pad short reviews, truncate long ones
- Use masking to ignore padding
- Use sequence models (LSTM/RNN) that handle variable input
- Current vectorization "solves" this by fixed-size output

### Q30: Why is 0.5 threshold appropriate for binary classification?
**A:**
- Sigmoid output: P(positive) ∈ [0, 1]
- 0.5 threshold: Natural decision boundary
- P ≥ 0.5: Predict positive (1)
- P < 0.5: Predict negative (0)
- Can adjust for different costs

---

## Comparison: MNIST vs IMDB

### Q31: Why does IMDB use more aggressive dropout (30%) than MNIST (20%)?
**A:**
- IMDB has fewer samples (25k train vs 60k for MNIST)
- Higher overfitting risk with less data
- More dropout needed for regularization
- Trade-off: Better generalization, less training accuracy

### Q32: Why Sigmoid for IMDB but Softmax for MNIST?
**A:**
- **Sigmoid**: Single output value (0-1) for binary
- **Softmax**: Multiple outputs, probability distribution
- Different loss functions: binary vs categorical cross-entropy
- Match output layer to task requirements

### Q33: How would you apply MNIST approach to 100-class problem?
**A:**
- Keep Softmax output (not Sigmoid)
- Change output neurons: 100 instead of 10
- Keep categorical cross-entropy loss
- One-hot encode labels to 100 values
- May need larger hidden layers

### Q34: How would you apply IMDB approach to multi-class sentiment?
**A:**
- Change output activation: Softmax (not Sigmoid)
- Multiple output neurons: 5 (for 5-star ratings)
- Change loss: Categorical cross-entropy
- One-hot encode 5 classes
- Rest stays same

### Q35: Why are these both "fully connected" (Dense) networks?
**A:**
- Every neuron in layer connects to every neuron in next layer
- Different from convolutional (spatial structure) or recurrent (temporal)
- MNIST could use Conv2D for better spatial understanding
- IMDB has no spatial/temporal structure (word order somewhat matters, but ignored here)

---

## Advanced Questions

### Q36: How would you implement early stopping?
**A:**
```python
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=5)
model.fit(x_train, y_train, callbacks=[early_stop], epochs=100, ...)
```
- Stops training if validation loss doesn't improve for 5 epochs
- Prevents overfitting, saves training time

### Q37: How would you apply data augmentation to MNIST?
**A:**
- Rotate images slightly (±15 degrees)
- Shift images by few pixels
- Add small noise
- Zoom in/out
- Increases training data without new annotations
- Use `ImageDataGenerator` in Keras

### Q38: Why would Conv2D layers be better for MNIST?
**A:**
- Conv2D respects spatial structure (nearby pixels related)
- Fewer parameters than Dense (weight sharing)
- Better feature extraction (edges, patterns)
- Faster training and less overfitting
- Typical benchmark: 99.2% with Conv2D vs 97.5% with Dense

### Q39: How would you use embedding layers for IMDB?
**A:**
```python
model = Sequential([
    Embedding(10000, 32, input_length=500),  # Maps word IDs to 32-dim vectors
    LSTM(32),
    Dense(1, activation='sigmoid')
])
```
- Converts word IDs to dense vectors
- Captures semantic relationships
- Better than one-hot encoding
- LSTM processes word sequences

### Q40: How would you visualize what the network learned?
**A:**
1. **Feature visualization**: Visualize filters in Conv layers
2. **Activation maximization**: Find inputs that maximize neurons
3. **Attention maps**: Show which input regions matter most
4. **t-SNE**: Visualize hidden layer embeddings
5. **CAM (Class Activation Map)**: Heatmap of important regions

---

## Debugging Questions

### Q41: What if training accuracy stays at 10% for MNIST?
**A:**
- Model learning nothing (random guessing is 10% = 1/10)
- Check: Did you normalize pixels (0-255 → 0-1)?
- Check: Is label encoding correct (one-hot)?
- Check: Is input shape correct (784)?
- Try: Simpler model, longer training, different optimizer

### Q42: What if test accuracy is 50% but training is 99% for IMDB?
**A:**
- Severe overfitting
- Model memorized training data
- Solutions: More dropout, fewer neurons, regularization
- Try: Reduce model size, train fewer epochs, more data

### Q43: What if model.fit() crashes with memory error?
**A:**
- Reduce batch_size: 128 → 64 → 32
- Reduce number of neurons: 512 → 256
- Reduce epochs if large dataset
- Use data generator to load batches
- Run on GPU if available

### Q44: What if predictions are all 0 or all 1?
**A:**
- Network collapsed to trivial solution
- Learning rate too high or too low
- Try different optimizer or learning rate
- Check data preprocessing
- Simpler model might converge better

### Q45: How do you know if model finished training properly?
**A:**
- Loss decreased overall (not monotonic, but trend down)
- Validation loss plateaued or started increasing (sign to stop)
- Accuracy increased to reasonable level (>95% for MNIST, >85% for IMDB)
- No error messages in training logs

---

## Practical Application Questions

### Q46: How would you deploy this MNIST model in production?
**A:**
1. Save model: `model.save('mnist_model.h5')`
2. Load in production: `model = tf.keras.models.load_model('mnist_model.h5')`
3. Preprocess new images same way (normalize)
4. Run inference: `prediction = model.predict(new_image)`
5. Return top class: `digit = np.argmax(prediction)`

### Q47: How would you handle new data in IMDB classifier?
**A:**
- Raw text review → tokenize → convert to word IDs
- Apply same vectorization as training
- Use same word index (don't recompute)
- Ensure 10,000 dimensions
- Run through model

### Q48: How would you create confidence scores?
**A:**
```python
predictions = model.predict(x_test)
# MNIST: np.max(predictions, axis=1) → max probability per sample
# IMDB: predictions[:, 0] → P(positive)

confidence = np.max(predictions, axis=1) * 100  # Convert to %
```
- High confidence: near 0 or 1
- Low confidence: near 0.5
- Can threshold: only accept if confidence > 90%

### Q49: How would you handle adversarial examples?
**A:**
1. **Detect**: Check prediction confidence
2. **Defend**: Adversarial training with augmented data
3. **Monitor**: Log low-confidence predictions
4. **Robust training**: Train with adversarial examples added
5. **Ensemble**: Vote of multiple models

### Q50: How would you update model with new data?
**A:**
```python
# Option 1: Fine-tune
model.fit(new_training_data, epochs=5, ...)  # Few epochs on new data

# Option 2: Retrain
new_model = build_new_model()
new_model.fit(all_data_combined, epochs=20, ...)
```
- Fine-tuning: Use pretrained weights, quick update
- Retraining: From scratch, more data needed but more thorough

---

## Summary Key Points

1. **Preprocessing is critical**: Normalization, reshaping, encoding
2. **Architecture matters**: Hidden layers, dropout, activation functions
3. **Loss and activation must match task**: Classification requires careful choices
4. **Regularization prevents overfitting**: Dropout, early stopping, validation
5. **Batch training enables efficient learning**: Gradient averaging
6. **Metrics guide model selection**: Accuracy, loss, validation curves
7. **Data matters most**: More data usually beats model engineering
8. **Generalization is the goal**: Test performance is what matters
9. **Monitoring prevents problems**: Validation curves detect issues early
10. **Save and deploy carefully**: Same preprocessing for inference as training
