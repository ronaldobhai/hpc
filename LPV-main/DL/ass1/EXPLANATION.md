# Assignment 1: Linear Regression vs Neural Networks - House Price Prediction

## Overview
This assignment compares traditional Linear Regression with a Neural Network for predicting house prices using the Boston Housing dataset. It demonstrates how deep learning models can outperform traditional ML approaches.

---

## Dataset: Boston House Price Data

### Dataset Loading
```python
boston = pd.read_csv("Boston House Price Data.csv")
boston.head()
```
- Loads the Boston Housing dataset from CSV file
- Contains features like LSTAT (% lower status population), RM (avg rooms per dwelling), PTRATIO (pupil-teacher ratio)
- Target variable: PRICE (median value of homes)

### Feature Selection
```python
X = boston[['LSTAT', 'RM', 'PTRATIO']]  # Selected features
y = boston['PRICE']                       # Target variable
```
- **LSTAT**: Percentage of lower status population - indicates neighborhood quality
- **RM**: Average number of rooms per dwelling - indicates house size
- **PTRATIO**: Pupil-teacher ratio by town - indicates school quality/neighborhood development
- Uses 3 important features instead of all 13 to simplify the model

### Train-Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
```
- Splits data into 80% training (412 samples) and 20% testing (103 samples)
- `random_state=4` ensures reproducibility
- Prevents information leakage - model hasn't seen test data during training

---

## Preprocessing: Feature Scaling

### Standard Scaler
```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Fit on training, then transform
X_test = scaler.transform(X_test)        # Transform test with same parameters
```

**Why Feature Scaling?**
- Different features have different ranges (LSTAT: 0-40, RM: 3-9, PTRATIO: 12-23)
- Neural Networks perform better when features are normalized to similar scales
- Speeds up gradient descent convergence

**How StandardScaler Works:**
- Formula: `(X - mean) / std_dev`
- Transforms data to have mean=0 and standard deviation=1
- Fit on training data only to avoid data leakage

---

## Model 1: Linear Regression

### Model Creation & Training
```python
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
```
- Creates a linear model: `PRICE = b0 + b1*LSTAT + b2*RM + b3*PTRATIO`
- Fits the model to training data by finding optimal coefficients
- Closed-form solution (no epochs, just one pass)

### Prediction
```python
y_pred_lr = lr_model.predict(X_test)
```
- Makes predictions on test set
- Linear model assumes linear relationship between features and price

### Evaluation Metrics
```python
mse_lr = mean_squared_error(y_test, y_pred_lr)    # Mean Squared Error
mae_lr = mean_absolute_error(y_test, y_pred_lr)   # Mean Absolute Error
r2_lr = r2_score(y_test, y_pred_lr)                # R-squared (coefficient of determination)
```

**Metrics Explained:**
- **MSE**: Average of squared errors, penalizes large errors more
  - Formula: `(1/n) * Σ(y_true - y_pred)²`
  - Lower is better
- **MAE**: Average absolute error, more interpretable
  - Formula: `(1/n) * Σ|y_true - y_pred|`
  - Lower is better
- **R²**: Proportion of variance explained (0-1 scale)
  - 1.0 = perfect fit, 0.0 = baseline, negative = worse than baseline

---

## Model 2: Neural Network

### Model Architecture
```python
model = Sequential([
    Dense(128, activation='relu', input_shape=(3,)),    # Input layer: 128 neurons
    Dense(64, activation='relu'),                        # Hidden layer 1: 64 neurons
    Dense(32, activation='relu'),                        # Hidden layer 2: 32 neurons
    Dense(16, activation='relu'),                        # Hidden layer 3: 16 neurons
    Dense(1)                                             # Output layer: 1 neuron (price)
])
```

**Architecture Breakdown:**
- **Input Layer**: 3 neurons (LSTAT, RM, PTRATIO)
- **Hidden Layers 1-3**: 128 → 64 → 32 → 16 neurons (decreasing pyramid)
- **Output Layer**: 1 neuron (predicted price)

**Why This Architecture?**
- Starts with many neurons (128) to capture complex patterns
- Gradually reduces (64 → 32 → 16) to avoid overfitting
- ReLU activation adds non-linearity for complex relationships

### ReLU Activation
```python
activation='relu'  # Rectified Linear Unit
```
- Formula: `f(x) = max(0, x)`
- Outputs 0 for negative values, identity for positive values
- Introduces non-linearity, allowing network to learn complex functions
- Computationally efficient and helps with gradient flow

### Compilation
```python
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
```

**Components:**
- **Optimizer 'adam'**: Adaptive learning rate, adjusts step size per parameter
  - Combines momentum (past gradients) with adaptive rates
  - Faster convergence than standard SGD
- **Loss 'mse'**: Mean Squared Error
  - Good for regression tasks
  - Penalizes large errors heavily
- **Metrics 'mae'**: Tracks Mean Absolute Error during training
  - Doesn't affect model learning, just for monitoring

### Training
```python
history = model.fit(X_train, y_train, 
                   epochs=100,              # Pass through entire dataset 100 times
                   batch_size=32,           # Process 32 samples at a time
                   validation_split=0.2,    # Use 20% of training for validation
                   verbose=1)               # Print progress
```

**Training Process:**
1. Forward pass: Input → predictions through network
2. Calculate loss using MSE
3. Backward pass: Compute gradients (backpropagation)
4. Update weights using Adam optimizer
5. Repeat for 100 epochs with validation monitoring

**Hyperparameters:**
- **epochs=100**: 100 complete passes through training data
- **batch_size=32**: Update weights after 32 samples (memory efficient)
- **validation_split=0.2**: Monitor performance on 20% held-out data
  - Helps detect overfitting if validation loss increases

### Prediction & Evaluation
```python
y_pred_nn = model.predict(X_test)
mse_nn, mae_nn = model.evaluate(X_test, y_test)  # Returns loss and metric
r2_nn = r2_score(y_test, y_pred_nn)
```

---

## Prediction on New Data

```python
new_data = np.array([[0.1, 10.0, 5.0]])           # 1 sample, 3 features
new_data_scaled = scaler.transform(new_data)      # Must scale same way as training
predicted_price = model.predict(new_data_scaled)  # Get prediction
print("Predicted Price:", predicted_price[0][0])
```

**Key Points:**
- Always transform new data using the fitted scaler from training
- Never refit scaler on new data (causes leakage)
- Prediction format: array of shape (1, 1) for single sample

---

## Comparison: Linear Regression vs Neural Network

| Aspect | Linear Regression | Neural Network |
|--------|---|---|
| **Complexity** | Simple linear equation | Deep non-linear model |
| **Interpretability** | Coefficients show feature impact | Black box, hard to interpret |
| **Training Speed** | Instant (closed-form) | Requires epochs (iterative) |
| **Non-linearity** | Cannot learn | ReLU layers enable learning |
| **Overfitting Risk** | Low (simple model) | High (complex, needs regularization) |
| **Performance** | Good baseline | Often better for complex relationships |

---

## Key Concepts

### Supervised Learning
- Model learns from labeled data (features → known targets)
- Learns mapping function to predict on new data

### Regression Task
- Predicting continuous values (prices)
- Different from classification (categories)

### Gradient Descent
- Iterative optimization algorithm
- Moves in direction of steepest descent to minimize loss
- Adam optimizer adapts learning rate per parameter

### Backpropagation
- Efficiently computes gradients through all layers
- Chain rule applied from output layer backward to input
- Enables training of deep networks

### Overfitting
- Model memorizes training data, poor generalization
- Validation split helps detect early
- Dropout layers (not used here) reduce overfitting

---

## Summary

This assignment demonstrates:
1. **Data preprocessing**: Scaling features for neural networks
2. **Model building**: Sequential API for regression
3. **Training**: Epochs, batch size, validation monitoring
4. **Evaluation**: MSE, MAE, R² metrics
5. **Comparison**: Advantages of deep learning over traditional ML
