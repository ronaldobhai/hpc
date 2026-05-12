# Assignment 1: QA - Linear Regression vs Neural Networks

---

## Conceptual Questions

### Q1: What is the main difference between Linear Regression and a Neural Network?
**A:** Linear Regression models relationships as linear equations (y = mx + b), while Neural Networks use multiple layers with non-linear activations (ReLU) to learn complex, non-linear relationships in data.

### Q2: Why do we use StandardScaler on the features?
**A:** StandardScaler normalizes features to have mean=0 and std=1. This helps neural networks:
- Converge faster during training (gradient descent works better with similar scales)
- Prevents features with larger ranges from dominating
- Ensures fair weight updates across all features

### Q3: What is the difference between training and test sets?
**A:** 
- **Training set (80%)**: Used to teach the model by adjusting weights
- **Test set (20%)**: Used ONLY to evaluate final performance
- Keeping them separate prevents overfitting (model memorizing training data)

### Q4: Why do we use validation_split=0.2 during training?
**A:** It uses 20% of training data as validation to monitor for overfitting. If validation loss increases while training loss decreases, the model is overfitting.

### Q5: What does the 'adam' optimizer do?
**A:** Adam adapts the learning rate for each parameter individually based on:
- Momentum: Information from previous gradients
- Adaptive rates: Different learning rates for different parameters
- Faster and more efficient than standard SGD

### Q6: Why is ReLU activation better than linear activation?
**A:** 
- **Linear activation**: Network becomes just a large linear equation (no improvement over Linear Regression)
- **ReLU activation**: `f(x) = max(0, x)` introduces non-linearity, allowing the network to learn complex curved relationships

### Q7: What does "backpropagation" mean?
**A:** Algorithm that computes gradients (derivatives) of loss with respect to each weight by:
1. Forward pass: Calculate predictions and loss
2. Backward pass: Compute gradients from output layer back to input layer using chain rule
3. Update weights in direction opposite to gradient (reduce loss)

### Q8: Why normalize predictions between 0 and 1 for neural networks?
**A:** 
- Neural networks with sigmoid/tanh activations output values in [0,1] or [-1,1]
- Input data in same range helps with numerical stability
- Prevents vanishing/exploding gradient problems

---

## Implementation Questions

### Q9: What would happen if we didn't scale the test data with the training scaler?
**A:** The test data would be on different scale than training data, causing:
- Poor model predictions (distribution mismatch)
- Violating the assumption the model was trained on
- Data leakage if we refit scaler on test data

### Q10: Why use mean_squared_error instead of mean_absolute_error for loss?
**A:** MSE:
- Penalizes large errors more (squared term)
- Smooth, differentiable everywhere (better for gradient descent)
- Encourages model to minimize outliers
- MAE is less sensitive to outliers but harder to optimize

### Q11: How would the model perform if we used only 1 or 2 features instead of 3?
**A:** 
- **Fewer features**: Model may underfit, can't capture all patterns
- **More features**: Might overfit, add noise
- **Current 3 features**: Balance between capturing relationships and avoiding overfitting

### Q12: What would happen if we increased epochs to 1000?
**A:** 
- Model would train longer
- Might overfit if validation loss stops improving
- Training loss would decrease further but test accuracy might worsen
- Good practice: monitor validation loss and use early stopping

### Q13: How does changing batch_size affect training?
**A:** 
- **Small batch (8)**: Noisier updates, might skip local minima, slower training
- **Large batch (128)**: Smoother updates, fewer steps, faster per-epoch but needs more memory
- **Default (32)**: Balance between stability and memory

### Q14: Why do we use Sequential() instead of other model types?
**A:** Sequential is simplest for:
- Linear stack of layers (input → layer1 → layer2 → output)
- Regression problems
- Simple architectures
- (Functional API needed for complex multi-input/output models)

### Q15: What would happen if we predicted on scaled data instead of unscaled?
**A:** 
- Predictions would be on different scale
- Model outputs would be hard to interpret
- If MSE loss was trained on scaled data, predictions match scaled space
- Must inverse transform to get original price range

---

## Mathematical Questions

### Q16: What is the formula for Mean Squared Error (MSE)?
**A:** `MSE = (1/n) * Σ(y_true_i - y_pred_i)²`
- n = number of samples
- Penalizes large errors quadratically
- Lower is better

### Q17: What is the formula for R² (R-squared)?
**A:** `R² = 1 - (SS_res / SS_tot)`
- `SS_res` = Σ(y_true - y_pred)²  (residual sum of squares)
- `SS_tot` = Σ(y_true - y_mean)²  (total sum of squares)
- Range: 0 to 1 (or negative for poor models)
- 0.8 = model explains 80% of variance

### Q18: What does the formula for StandardScaler look like?
**A:** For each feature:
```
x_scaled = (x - mean) / std_dev
```
- Subtracts mean (centers at 0)
- Divides by std dev (scales to unit variance)
- Inverse: `x_original = (x_scaled * std_dev) + mean`

### Q19: How is the ReLU activation function defined mathematically?
**A:** `f(x) = max(0, x)`
- For positive x: output = x
- For negative x: output = 0
- Derivative: `f'(x) = 1 if x > 0 else 0`

### Q20: What is the Dense layer transformation?
**A:** `y = activation(W @ x + b)`
- W: weight matrix (input_size × output_size)
- x: input vector
- b: bias vector
- activation: ReLU or other function
- @ denotes matrix multiplication

---

## Practical Application Questions

### Q21: How would you improve the Linear Regression model?
**A:**
1. Add polynomial features (x², x³) for non-linearity
2. Use more features from dataset
3. Use regularization (Ridge, Lasso) to prevent overfitting
4. Feature engineering (create new meaningful features)

### Q22: How would you improve the Neural Network model?
**A:**
1. Add Dropout layers for regularization
2. Use batch normalization for stable training
3. Add L1/L2 regularization to loss function
4. Hyperparameter tuning (more/fewer layers, neurons, learning rate)
5. Data augmentation if applicable
6. Collect more training data

### Q23: What if the test MSE is very low but training MSE is high?
**A:** This is unusual but might mean:
- The test set is easier than training set
- Random chance (test set has samples easier to predict)
- Check for data leakage
- Could happen with small test sets (sampling variance)

### Q24: What if training MSE and test MSE are both high?
**A:** Model underfits - it's not learning well:
- Try more complex model (more layers/neurons)
- Train longer (more epochs)
- Improve features (feature engineering)
- Check data quality (missing values, outliers)

### Q25: What if training MSE is low but test MSE is high?
**A:** Classic overfitting - model memorizes training data:
- Add Dropout layers
- Use L1/L2 regularization
- Train fewer epochs (early stopping)
- Use simpler model
- Collect more training data

---

## Design Questions

### Q26: How would you handle missing values in the dataset?
**A:**
- Remove rows with missing values (if few)
- Impute with mean/median/mode
- Use forward fill for time series
- Predictive imputation with other features
- Mark missing as separate category

### Q27: How would you handle outliers in house prices?
**A:**
1. **Detect**: IQR method (1.5×IQR) or Z-score (>3σ)
2. **Handle**:
   - Remove if data entry error
   - Keep if real (e.g., luxury homes)
   - Cap/floor at percentiles
   - Use MAE loss instead of MSE (less sensitive)
3. **Prevent**: StandardScaler makes model robust to some outliers

### Q28: How would you select which features to use?
**A:**
- **Correlation analysis**: Compute correlation with target
- **Variance**: Remove features with near-zero variance
- **Domain knowledge**: Use features that make sense
- **Feature importance**: From tree-based models
- **Recursive feature elimination**: Iteratively remove worst features
- **Dimensionality reduction**: PCA to combine features

### Q29: How would you handle imbalanced data (if this were classification)?
**A:**
- **Undersampling**: Remove majority class samples
- **Oversampling**: Duplicate minority class samples
- **SMOTE**: Generate synthetic minority samples
- **Class weights**: Penalize misclassifying minority class more
- **Different metrics**: Use F1-score, precision-recall instead of accuracy

### Q30: What would you change to apply this to a different dataset?
**A:**
1. Change CSV filename in `pd.read_csv()`
2. Update feature selection (X columns) based on new dataset
3. Adjust scaler if new features have different ranges
4. Modify network architecture (neurons, layers) based on feature count
5. Retune hyperparameters (epochs, batch_size)
6. Adjust validation split and test split based on dataset size
7. Update evaluation metrics if classification task

---

## Debugging Questions

### Q31: What if model.fit() throws a shape mismatch error?
**A:** Check that:
- Input shape matches `input_shape=(3,)` in first Dense layer
- Training data X has 3 features, y has 1 column
- Both X and y have same number of rows
- After reshaping/scaling, check `.shape` attribute

### Q32: What if predictions are all the same value?
**A:** 
- Model failed to learn patterns
- Learning rate too low (weights barely update)
- Network too simple (can't capture relationships)
- Data preprocessing issue (all features same scale but different meaning)
- Try: more layers, more neurons, more epochs, better features

### Q33: What if loss becomes NaN during training?
**A:**
- Learning rate too high (exploding gradients)
- Data has NaN/infinity values
- Numerical instability in loss calculation
- Fix: Lower learning rate, check data quality, use gradient clipping

### Q34: How do you know if the model is overfitting?
**A:**
- Training loss decreasing, validation loss increasing/plateauing
- Training MSE very low but test MSE high
- Model performs well on training data, poorly on new data
- Check loss curves from `history` object

### Q35: What if the model trains very slowly?
**A:**
- Dataset too large (reduce or use batching)
- Network too deep (too many layers)
- Learning rate too low (slow convergence)
- Batch size too small (more gradient steps)
- Try: larger batch_size, fewer layers, higher learning_rate, GPU

---

## Extension Questions

### Q36: How would you ensemble Linear Regression and Neural Network predictions?
**A:**
- Simple average: `(y_pred_lr + y_pred_nn) / 2`
- Weighted average: `0.4*y_pred_lr + 0.6*y_pred_nn` (based on performance)
- Stacking: Train meta-learner on both predictions
- Usually better than single model

### Q37: How would you use cross-validation for this problem?
**A:**
- K-Fold CV: Split data into k folds
- Train on k-1 folds, test on 1 fold
- Repeat k times, average results
- Gives robust estimate of model performance
- Better than single train-test split

### Q38: How would you apply this to time series (e.g., stock price)?
**A:**
- Don't shuffle (maintain temporal order)
- Walk-forward validation: train on past, test on future
- Use sliding window instead of random split
- Add temporal features (day-of-week, seasonality)
- Consider autoregressive models (ARIMA, LSTM)

### Q39: How would you add regularization to the Neural Network?
**A:**
```python
# L2 regularization (Ridge)
Dense(64, activation='relu', kernel_regularizer=l2(0.01))

# Dropout layer (75% of neurons stay)
Dropout(0.25)

# Both together
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.2))
```
- Reduces overfitting by penalizing large weights

### Q40: What advanced architectures could you use for regression?
**A:**
1. **ResNets**: Skip connections prevent vanishing gradients
2. **Attention**: Weights inputs based on importance
3. **Autoencoders**: Learn compressed representations
4. **Ensemble**: Multiple networks voting
5. **Mixture of Experts**: Different networks for different regions
6. **Bayesian Neural Networks**: Uncertainty estimation

---

## Summary of Key Learning Points

1. **Preprocessing is crucial** - Scaling, train-test split, feature engineering
2. **Neural Networks learn non-linear relationships** that Linear Regression cannot
3. **Validation during training prevents overfitting** 
4. **Adam optimizer efficiently updates weights** with adaptive learning rates
5. **ReLU activations enable network depth** through non-linearity
6. **Metrics (MSE, MAE, R²) help evaluate** model performance
7. **Data leakage risks** when scaling or transforming improperly
8. **Hyperparameter tuning** (layers, neurons, epochs) improves results
9. **Trade-offs exist** between model complexity and generalization
10. **Always validate on unseen test data** for honest performance assessment
