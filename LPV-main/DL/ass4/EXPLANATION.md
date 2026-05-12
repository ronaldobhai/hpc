# Assignment 4: Recurrent Neural Networks - Stock Price Prediction with LSTM

## Overview
This assignment predicts Google (GOOGL) stock prices using Long Short-Term Memory (LSTM) networks. LSTM is a type of Recurrent Neural Network (RNN) that learns patterns in sequential time-series data.

---

## Dataset Loading & Exploration

### Data Import
```python
import pandas as pd
data = pd.read_csv("GOOGL.csv", parse_dates=['Date'])
data = data.sort_values('Date')  # Ensure chronological order
```
- **GOOGL.csv**: Google stock price history
- **parse_dates=['Date']**: Converts date strings to datetime objects
- **sort_values('Date')**: Orders by date (important for time series!)

### Feature Selection
```python
features = ['Open', 'High', 'Low', 'Close', 'Volume']
data_full = data[features].values  # Shape: (num_days, 5)
```

**Features Explanation:**
- **Open**: Price at market open
- **High**: Highest price of the day
- **Low**: Lowest price of the day
- **Close**: Price at market close (predict this)
- **Volume**: Number of shares traded

**Why multiple features?**
- LSTM learns patterns from all features, predicts Close
- Single feature (Close only) less informative
- Example: High volume + high price might indicate bullish trend

### Train-Test Split
```python
train_size = int(len(data_full) * 0.8)  # 80% training
train_data = data_full[:train_size]     # Earlier dates (past)
test_data = data_full[train_size:]      # Later dates (future)
```

**Critical for Time Series:**
- **Never shuffle time series data**
- Always split by time: train on past, test on future
- Prevents information leakage (can't peek at future)

---

## Data Preprocessing

### Scaling with MinMaxScaler
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_data)     # Fit on train
test_scaled = scaler.transform(test_data)           # Use same scaler
```

**Why MinMaxScaler?**
- Scales values to [0, 1] range
- Formula: `(x - min) / (max - min)`
- Neural networks work better with normalized data
- Prevents large values from dominating gradients

**Why fit only on train data?**
- Prevents data leakage: test statistics shouldn't influence training
- Use training min/max for test scaling
- Inverse transformation: `scaled_value * (max - min) + min`

---

## Creating Sequences (Windowing)

### Problem: Variable-Length Sequences
```python
# Raw time series: [p1, p2, p3, p4, p5, ...]
# Can't directly input to neural networks (variable length)
# Need fixed-size windows
```

### Solution: Create Sliding Windows
```python
def create_sequences(data, window_size=60):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])        # 60 previous days
        y.append(data[i, 3])                   # Today's close (index 3)
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_scaled, window_size=60)
```

**How it works:**
```
Data: [day1, day2, day3, ..., day60, day61, day62, ...]

Sequence 1: X = [day1-day60], y = day61 (close price)
Sequence 2: X = [day2-day61], y = day62 (close price)
Sequence 3: X = [day3-day62], y = day63 (close price)
...
```

**Window Size = 60:**
- Uses 60 previous days to predict next day
- Captures 3 months of history
- Longer window: more history, more computation
- Shorter window: faster, less history

**Output Shapes:**
- **X_train**: (num_sequences, 60, 5) - 60 days × 5 features
- **y_train**: (num_sequences,) - single close price to predict

### Test Sequences
```python
last_train_days = train_scaled[-window_size:]  # Last 60 training days
full_test_input = np.vstack((last_train_days, test_scaled))  # Combine
X_test, y_test = create_sequences(full_test_input, window_size)
```

**Why combine train and test?**
- Model needs 60-day window to start predicting
- Can't use 60 test days (they're the future)
- Use last 60 train days + test days
- Ensures smooth transition from train to test

---

## LSTM Model Architecture

### LSTM Layer Explanation
```python
model = Sequential([
    LSTM(units=60, return_sequences=True, 
         input_shape=(X_train.shape[1], X_train.shape[2])),  # (60, 5)
    Dropout(0.2),
    
    LSTM(units=60, return_sequences=False),
    Dropout(0.2),
    
    Dense(units=25),
    Dense(units=1)
])
```

### What is LSTM?
Long Short-Term Memory - a type of RNN that:
- **Remembers** long-term dependencies
- **Forgets** irrelevant information
- Uses gates to control information flow

**Three Gates:**
1. **Forget Gate**: Decide which information to discard
2. **Input Gate**: Decide which new information to add
3. **Output Gate**: Decide what to output

**Why LSTM over vanilla RNN?**
- Vanilla RNN: Gradients vanish over long sequences
- LSTM: Gates control gradient flow, learning over longer periods
- Better for time series with 60+ step dependencies

### Layer 1: LSTM(units=60, return_sequences=True)
```python
LSTM(units=60, return_sequences=True, input_shape=(60, 5))
```

**Parameters:**
- **units=60**: 60 LSTM cells (internal memory units)
- **return_sequences=True**: Output all timesteps, not just last
- **input_shape=(60, 5)**: 60 timesteps × 5 features

**Input/Output:**
- Input: (batch_size, 60, 5) - 60 days × 5 features
- Output: (batch_size, 60, 60) - 60 days × 60 LSTM units

**return_sequences=True meaning:**
- Returns output for all 60 timesteps
- Next LSTM layer processes each timestep
- Allows stacking multiple LSTMs

### Layer 2: Dropout
```python
Dropout(0.2)  # Drop 20% of LSTM outputs
```
- Prevents overfitting
- Randomly deactivates 20% of LSTM outputs during training

### Layer 3: LSTM(units=60, return_sequences=False)
```python
LSTM(units=60, return_sequences=False)
```

**Key Difference:**
- **return_sequences=False**: Returns only last timestep's output
- Input: (batch_size, 60, 60) from previous layer
- Output: (batch_size, 60) - just the final LSTM output

**Why return_sequences=False here?**
- Processes 60-day history through second LSTM
- Outputs single summary representation
- This represents learned pattern from 60 days
- Can then use for prediction (Dense layer)

### Layer 4: Dropout
```python
Dropout(0.2)
```
- More regularization before Dense layers

### Layer 5: Dense(units=25)
```python
Dense(units=25)
```
- Takes 60-dimensional LSTM output
- Learns 25 abstract patterns
- Could be "trend", "volatility", "support levels", etc.

### Layer 6: Dense(units=1)
```python
Dense(units=1)
```
- Final output: single value (predicted close price)
- No activation function (regression, output any range)
- Inverse transformed to get actual price

---

## Compilation & Training

### Compilation
```python
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)
```

**MSE Loss:**
- Typical for regression/time series
- Penalizes large prediction errors

**Adam Optimizer:**
- Adaptive learning rates per parameter
- Good for complex models like LSTM

### Training Considerations
- 20 epochs × multiple batches per epoch
- Monitors training loss
- No validation shown in code (could add)

---

## Prediction & Inverse Transformation

### Making Predictions
```python
predictions = model.predict(X_test)  # Shape: (num_test_sequences, 1)
```
- Model outputs scaled values [0, 1]
- Need to convert back to actual prices

### Inverse Transform to Original Scale
```python
# Repeat predictions to match feature count (5 features)
prediction_copies = np.repeat(predictions, len(features), axis=-1)
y_pred_final = scaler.inverse_transform(prediction_copies)[:, 3]
```

**Why repeat?**
- Predictions: (num_sequences, 1)
- Need: (num_sequences, 5) for inverse_transform
- Copy prediction to all 5 features, then inverse transform
- Extract column 3 (Close price) which we predicted

**Example:**
```
Prediction (scaled): [0.6]
Repeated: [[0.6, 0.6, 0.6, 0.6, 0.6]]
After inverse_transform: [[40, 50, 30, 45, 1000000]]
Extract close [3]: 45
```

### Scaling Test Labels Similarly
```python
test_copies = np.repeat(y_test.reshape(-1, 1), len(features), axis=-1)
y_test_final = scaler.inverse_transform(test_copies)[:, 3]
```
- Same process for actual values
- Enables fair comparison with predictions

---

## Why LSTM for Stock Prices?

### Advantages
1. **Sequential Data**: LSTM designed for time series
2. **Long-term Dependencies**: Remembers patterns over 60 days
3. **Adaptivity**: LSTM gates adjust what to remember
4. **Non-linear**: Captures complex market patterns

### Challenges
1. **Market Randomness**: Some price movements random
2. **External Factors**: News, events not in data
3. **Non-stationary**: Market regime changes over time
4. **Small Prediction Window**: Predicting 1 day ahead difficult

### Alternative Approaches
- Exponential Smoothing: Simpler, good baseline
- ARIMA: Traditional time series
- Prophet: Facebook's time series library
- Transformer: Attention-based (newer than LSTM)

---

## Key Time Series Concepts

### Stationarity
- Time series with constant mean/variance
- Stock prices non-stationary (trend up/down)
- Differences (price_today - price_yesterday) more stationary
- Could use differencing before LSTM

### Autocorrelation
- Price today correlated with past prices
- LSTM learns these correlations
- 60-day window captures relevant correlations

### Seasonal Patterns
- Stock might have weekly/monthly patterns
- LSTM can learn if patterns consistent
- More data helps learning patterns

### Trend
- Stock follows overall trend (up or down)
- LSTM should learn trend component
- Longer windows help capture long-term trends

---

## Summary of Key Concepts

1. **Time Series**: Data ordered by time, don't shuffle
2. **Windowing**: Create fixed-size sequences from variable-length data
3. **LSTM**: RNN architecture for sequential learning
4. **Gates**: Forget/Input/Output gates control information flow
5. **return_sequences**: Whether to output all timesteps or last
6. **Stacking LSTMs**: Multiple layers learn hierarchical patterns
7. **Inverse Transform**: Convert predictions back to original scale
8. **Data Leakage**: Test data can't influence training
9. **Prediction Challenges**: Markets are partly random
10. **Evaluation**: Compare predictions to actual prices
