# Assignment 4: QA - Recurrent Neural Networks and LSTM for Time Series

---

## Conceptual RNN Questions

### Q1: What is a Recurrent Neural Network (RNN)?
**A:** Neural network that processes sequences by:
- Taking one element at a time (word, price, etc.)
- Maintaining hidden state (memory of previous elements)
- Updating state based on current input and previous state
- Can process variable-length sequences

### Q2: What is LSTM and why is it better than vanilla RNN?
**A:**
- **Vanilla RNN**: Simple recurrent connections, gradients vanish over long sequences
- **LSTM**: Special architecture with gates controlling information flow
- **Advantages**:
  - Long-term dependencies: Remembers 60+ steps back
  - Gradient flow: Gates prevent vanishing gradients
  - Selective memory: Forgets irrelevant, remembers important

### Q3: What are the three gates in LSTM?
**A:**
1. **Forget Gate**: Decides which old information to discard
   - Output: Weights applied to previous cell state
   - Example: "Forget yesterday's volatility, high changes"
2. **Input Gate**: Decides what new information to add
   - Output: Weights for new candidate values
   - Example: "Add today's price trend"
3. **Output Gate**: Decides what to output
   - Output: Final hidden state for next step
   - Example: "Output aggregated trend signal"

### Q4: Why use windowing (60-day sequences) instead of processing each day individually?
**A:**
- LSTM needs context (history) to predict
- Single day insufficient: Stock price depends on trend
- 60 days: Captures 3-month pattern
- Provides temporal context for LSTM to learn from

### Q5: What is return_sequences=True vs return_sequences=False?
**A:**
- **return_sequences=True**: Output shape (batch, steps, units)
  - Returns all timesteps' outputs
  - Used when feeding to another LSTM/Dense
  - Example: (32, 60, 60) for 60 timesteps × 60 LSTM units
- **return_sequences=False**: Output shape (batch, units)
  - Returns only last timestep's output
  - Used before Dense layer for prediction
  - Example: (32, 60) just the final output

### Q6: Why train on past data and test on future data (not shuffle)?
**A:**
- Stock prices ordered by time (temporal structure)
- Shuffling destroys order, violates causality
- Can't train on future to predict past
- Must: Train on past → Predict future
- Information leakage if test data used in training

### Q7: What does Dropout do in LSTM context?
**A:**
- Randomly deactivates neurons during training (20% here)
- Prevents LSTM from over-relying on specific cells
- Forces network to learn robust representations
- Reduces co-adaptation between cells
- Improves generalization to test data

### Q8: What is the difference between MSE and MAE for stock prediction?
**A:**
| Metric | Formula | Properties |
|--------|---------|---|
| MSE | Σ(y_true - y_pred)² / n | Penalizes large errors more, squared units |
| MAE | Σ\|y_true - y_pred\| / n | Linear penalty, same units as price |
- **MSE (current)**: Useful for volatile prices, captures outliers
- **MAE**: More interpretable (average error in dollars)
- **RMSE**: √MSE, back to original units (good compromise)

### Q9: Why use MinMaxScaler [0, 1] instead of StandardScaler [mean=0, std=1]?
**A:**
- **MinMax [0, 1]**: Bounded range, works better with LSTM activations
- **StandardScaler**: Unbounded, can be negative
- **For inverse transform**: MinMax easier to reverse (multiply/divide)
- **For sigmoid/tanh**: MinMax [0, 1] natural output range
- Both work, MinMax conventional for time series

### Q10: What would happen if you didn't normalize data?
**A:**
- Large raw values (stock price ~$100) cause large gradients
- Training unstable: weights oscillate wildly
- Slower convergence or divergence
- Numerical issues: overflow/underflow
- Model biased toward large-scale features

---

## Implementation Questions

### Q11: Why fit scaler only on training data?
**A:**
- **Min/max values**: Should represent training distribution
- **If fit on test too**: Test information leaks into training
- **Test scaling**: Use training min/max, not test min/max
- **Inverse transform**: Use same scaler that fitted training
- Critical to prevent overly optimistic results

### Q12: How would you handle different time periods (hourly vs daily)?
**A:**
- **Daily** (current): 60 days window
- **Hourly**: 60 × 24 = 1440 hour window (same 2.5 days)
- **Weekly**: 60 / 5 = 12 week window (3 months)
- Window_size is hyperparameter, adjust for granularity
- More frequent data might need longer window

### Q13: What if you changed window_size from 60 to 30?
**A:**
- **Less history**: Model sees only 1 month
- **Faster training**: Fewer sequences to process
- **Fewer parameters**: Shorter input dimension
- **Risk**: Misses longer-term patterns
- **Trade-off**: Speed vs. learning capacity

### Q14: What if you changed window_size to 180?
**A:**
- **More history**: 6 months of patterns
- **Slower training**: More sequences, longer computation
- **Risk**: Outdated information (6 months old)
- **Benefit**: Captures longer cycles (seasons, yearly)
- **Good if**: Long-term trends dominant

### Q15: How would you handle missing data in stock prices?
**A:**
- **Forward fill**: Use last known price
- **Interpolation**: Linear interpolate between values
- **Removal**: Remove rows with missing values (if few)
- **Indicator**: Create binary column marking missing
- **Note**: Stock market closed weekends/holidays naturally

### Q16: What would happen if volume column had NaN values?
**A:**
```python
# Detect NaN
if np.isnan(data_full).any():
    print("NaN values found")
    
# Handle options:
data_full = data_full.fillna(method='ffill')  # Forward fill
data_full = data_full.fillna(data_full.mean())  # Fill with mean
```
- NaN breaks LSTM training (gradient computation fails)
- Must handle before creating sequences

### Q17: How would you predict multiple days ahead?
**A:**
```python
# Current: Predict day_i using days [i-60:i]

# Multiple days:
# Option 1: Predict day_i, day_i+1, day_i+2 separately
# Option 2: Output Dense(3) instead of Dense(1)
# Option 3: Autoregressive: Use prediction as input

y_train: shape (num_sequences, 3)  # Next 3 days
model.fit(X_train, y_train, ...)
```
- More complex: Predictions drift further into future

### Q18: How would you predict multiple features (Open, Close, etc.)?
**A:**
```python
# Current: Predict only Close

# Multi-output:
Dense(5)  # Predict all 5 features

y_train: shape (num_sequences, 5)  # All features
model.fit(X_train, y_train, ...)
```
- Shared learning across features
- Might improve accuracy through multi-task learning

### Q19: What does batch_size=32 mean during training?
**A:**
- Process 32 sequences at a time
- Update weights after processing 32 sequences (not 32 days)
- Trade-off: 
  - Larger batch: More stable, fewer updates per epoch
  - Smaller batch: Noisier, more updates (might escape local minima)

### Q20: How would you use model.evaluate() on test data?
**A:**
```python
test_loss = model.evaluate(X_test, y_test)
print("Test loss (MSE):", test_loss)

# Then make predictions
predictions = model.predict(X_test)
```
- evaluate(): Returns loss value
- predict(): Returns predictions
- Can compare: (y_test_final vs y_pred_final)

---

## Time Series Specific Questions

### Q21: What is the difference between train-test split for images vs time series?
**A:**
| Aspect | Images | Time Series |
|--------|--------|---|
| **Order** | No order, can shuffle | Order critical, chronological |
| **Split** | Random 80-20 | Sequential: first 80%, last 20% |
| **Validation** | Cross-validation works | Time-based split only |
| **Leakage** | OK if random | NOT OK if future used in training |

### Q22: What is "information leakage" in time series?
**A:**
- Using future information to predict past
- Example: Train on [day1-day100, day101-day200], test on day150
- Day150 in test but nearby dates in training
- Causes optimistic performance estimates
- Always: Early dates train, later dates test

### Q23: What would walk-forward validation look like?
**A:**
```python
# More realistic for time series
for split_date in date_splits:
    train = data[:split_date]
    test = data[split_date:split_date+30]
    model.fit(train)
    evaluate(test)
    
# Simulates real deployment: Train on past, predict future
```
- More robust than single train-test split
- More computationally expensive

### Q24: What is stationarity and why does it matter?
**A:**
- **Stationary**: Mean, variance, autocorrelation constant over time
- **Non-stationary**: Stock prices (trend up/down, changing volatility)
- **Why matters**: LSTM assumes patterns repeat
- **Solution**: Differencing (today - yesterday) becomes stationary
- **Alternative**: LSTM might learn non-stationary patterns anyway

### Q25: What is autocorrelation?
**A:**
- Correlation between time series and itself at different lags
- Price today correlated with price 1 day ago
- Autocorr at lag-1: corr(price[t], price[t-1])
- Autocorr at lag-60: corr(price[t], price[t-60])
- LSTM learns autocorrelation patterns automatically

---

## Performance & Evaluation

### Q26: What metrics would you use to evaluate stock prediction?
**A:**
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

mse = mean_squared_error(y_test_final, y_pred_final)
mae = mean_absolute_error(y_test_final, y_pred_final)
rmse = np.sqrt(mse)

# Time series specific:
# MAPE: Mean Absolute Percentage Error
mape = np.mean(np.abs((y_test_final - y_pred_final) / y_test_final)) * 100
```
- **MSE**: Penalizes large errors
- **MAE**: Interpretable in dollars
- **RMSE**: Back to original units
- **MAPE**: Percentage error (accounts for scale)

### Q27: What if MAE is $5 - is that good?
**A:**
- **Depends on**: Stock price level
- Price $100: $5 error = 5% (good)
- Price $20: $5 error = 25% (poor)
- **Relative** (MAPE) more informative than absolute (MAE)
- Compare to baseline: Using last price? Average? Expert prediction?

### Q28: How would you compare to baseline models?
**A:**
```python
# Baseline 1: Naive (tomorrow same as today)
baseline_pred = y_test_final[:-1]  # Shift by 1

# Baseline 2: Last week's price
baseline_pred = y_test_final[-7:]

# Baseline 3: Average recent price
baseline_pred = np.mean(y_test_final[-30:])

# LSTM only valuable if better than baselines
if lstm_mae < baseline_mae:
    print("LSTM useful!")
```

### Q29: What if training loss decreases but test MSE increases?
**A:**
- Clear overfitting: Model memorizes training data
- Solutions:
  1. More Dropout (0.2 → 0.3)
  2. Simpler model (fewer LSTM units: 60 → 30)
  3. More data (if available)
  4. Early stopping (stop when test starts increasing)
  5. Regularization: L1/L2 penalty on weights

### Q30: How would you detect if model is underfitting?
**A:**
- Both training and test loss stay high
- Model not learning patterns
- Solutions:
  1. More LSTM units: 60 → 120
  2. More layers: Add another LSTM
  3. More epochs: 20 → 50
  4. Better features: Include other indicators
  5. Longer window: 60 → 90 days

---

## Advanced Architecture Questions

### Q31: How would you stack 3 LSTM layers?
**A:**
```python
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(60, 5)),
    Dropout(0.2),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])
```
- First two: return_sequences=True (pass to next LSTM)
- Last one: return_sequences=False (only final output)
- Deeper = more parameters, more learning capacity
- Risk: Harder to train, more overfitting

### Q32: What is Bidirectional LSTM?
**A:**
```python
from tensorflow.keras.layers import Bidirectional

model = Sequential([
    Bidirectional(LSTM(60, return_sequences=True), input_shape=(60, 5)),
    ...
])
```
- Processes sequence forward AND backward
- Can see future context (better for time series)
- Problem for real prediction: Need future data to see backward
- Good for analysis, not deployment prediction

### Q33: How would you use Attention mechanism?
**A:**
```python
from tensorflow.keras.layers import LSTM, Attention, AdditiveAttention

# Attention weights which timesteps matter
model = Sequential([
    LSTM(60, return_sequences=True, input_shape=(60, 5)),
    AdditiveAttention(),  # Learn which days important
    LSTM(60, return_sequences=False),
    Dense(25),
    Dense(1)
])
```
- Learns which timesteps to focus on
- More sophisticated than fixed 60-day window
- Can ignore irrelevant days

### Q34: What is 1D Convolution for time series?
**A:**
```python
from tensorflow.keras.layers import Conv1D

model = Sequential([
    Conv1D(32, 3, activation='relu', input_shape=(60, 5)),
    # Detects patterns in 3-day windows
    LSTM(60),
    Dense(1)
])
```
- Conv1D: Filters sliding over time (like Conv2D over space)
- Fast pattern detection before LSTM
- Alternative to stacked LSTMs
- Good for capturing quick patterns

### Q35: How would you use ensemble methods?
**A:**
```python
# Train multiple models
model1 = create_lstm_model()
model2 = create_lstm_model()
model3 = create_lstm_model()

# Different initializations, learn different patterns

pred1 = model1.predict(X_test)
pred2 = model2.predict(X_test)
pred3 = model3.predict(X_test)

ensemble_pred = (pred1 + pred2 + pred3) / 3
```
- Ensemble usually better than individual models
- Reduces variance, increases robustness

---

## Real-World Application

### Q36: How would you deploy this model for live trading?
**A:**
1. **Save model**: `model.save('price_predictor.h5')`
2. **Real-time data**: Fetch latest stock data
3. **Prepare input**: Last 60 days → scale → reshape to (1, 60, 5)
4. **Predict**: `prediction = model.predict(input_data)`
5. **Inverse transform**: Get actual price
6. **Decision logic**: If pred > current, buy? (Your strategy)

### Q37: How would you handle concept drift (market regime changes)?
**A:**
- Markets change: Bull→Bear, Volatility↓↑, Correlations shift
- Solutions:
  1. Retrain periodically (monthly/quarterly)
  2. Online learning: Update with new data
  3. Multiple models for different regimes
  4. Adaptive algorithms: Detect change, switch models

### Q38: How would you incorporate other features?
**A:**
```python
# Current: Open, High, Low, Close, Volume
# Add:
features = ['Open', 'High', 'Low', 'Close', 'Volume',
            'MA_20',      # 20-day moving average
            'Volatility', # Rolling std dev
            'RSI',        # Relative Strength Index
            'MACD']       # Moving Average Convergence Divergence

# Technical indicators improve predictions
```
- Feature engineering often more important than model
- Domain knowledge (finance) helps

### Q39: How would you estimate prediction confidence?
**A:**
```python
# Use ensemble
predictions = [m.predict(X_test) for m in models]
std_dev = np.std(predictions, axis=0)

# Or Bayesian DL: Predict with uncertainty
# Or Monte Carlo Dropout: Run inference multiple times with dropout

confidence = 1 / (std_dev + epsilon)  # Higher std = lower confidence
```

### Q40: What are limitations of this approach?
**A:**
1. **Past ≠ Future**: Historical patterns may not repeat
2. **Black swan events**: Unexpected shocks
3. **External factors**: News, politics not in data
4. **Efficient markets**: If it works, price already adjusted
5. **Overfit risk**: Complex model fits noise
6. **Transaction costs**: Profit margin might be small
7. **Implementation**: Real trading has slippage, latency

---

## Debugging Questions

### Q41: What if predictions are all close to mean price?
**A:**
- Model learned trivial solution (output average)
- Not learning temporal patterns
- Check: Did you create sequences correctly?
- Check: Is data normalized?
- Try: More LSTM units, more layers, more epochs

### Q42: What if model crashes with "shapes not aligned"?
**A:**
- Dimension mismatch somewhere
- Check: X_train shape == (samples, 60, 5)
- Check: y_train shape == (samples,) or (samples, 1)
- Print shapes: `print(X_train.shape, y_train.shape)`
- Verify create_sequences() output

### Q43: What if test predictions are NaN?
**A:**
- Model weights contain NaN/infinity
- Training crashed during learning
- Check: Learning rate too high?
- Check: Gradient explosions (LSTM prone)?
- Try: Add gradient clipping: `optimizer=Adam(clipvalue=1.0)`

### Q44: What if model won't converge (loss stays high)?
**A:**
- Not learning anything meaningful
- Check: Data properly scaled [0, 1]?
- Check: window_size big enough (60 good)?
- Try: Larger learning rate (Adam adaptive helps)
- Try: Simpler model first (fewer LSTM units)

### Q45: How to tell if model is overfitting?
**A:**
```python
history = model.fit(..., validation_split=0.1)
# Plot training vs validation loss

import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.legend()
plt.show()

# If val_loss increases while train decreases = overfitting
```

---

## Summary Key Points

1. **Time series**: Don't shuffle, chronological train-test split
2. **Windowing**: Create fixed sequences from continuous data
3. **LSTM**: Handles sequential dependencies through gates
4. **Scaling**: MinMax [0, 1] for neural networks
5. **Data leakage**: Catastrophic if future used in training
6. **return_sequences**: Critical for stacking LSTMs
7. **Inverse transform**: Convert predictions back to original units
8. **Baseline comparison**: LSTM only valuable if better than naive
9. **Overfitting risk**: Dropout, regularization important
10. **Real-world challenges**: Market isn't stationary, patterns change
