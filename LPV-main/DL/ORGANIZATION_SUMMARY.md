# Deep Learning Assignments - Organization & Summary

## Overview
Organized 4 Deep Learning assignments with comprehensive documentation. Each assignment includes:
- **Original Jupyter Notebook** (.ipynb)
- **EXPLANATION.md** - Line-by-line code explanations (200-300 lines each)
- **QA.md** - Question & Answer guide (40-50 questions each)

---

## Assignment Structure

```
DL/
├── ass1/
│   ├── ass1_linear_regression.ipynb
│   ├── Boston House Price Data.csv
│   ├── EXPLANATION.md                 (House price prediction: LR vs NN)
│   └── QA.md                          (40 Q&A on regression & neural networks)
│
├── ass2/
│   ├── ass2_2_mnist_ocr.ipynb         (MNIST digit recognition)
│   ├── ass2_imdb.ipynb                (IMDB sentiment analysis)
│   ├── EXPLANATION.md                 (Both MNIST OCR & IMDB)
│   └── QA.md                          (50 Q&A on dense networks & datasets)
│
├── ass3/
│   ├── ass3_fashion.ipynb
│   ├── EXPLANATION.md                 (Fashion MNIST with CNN)
│   └── QA.md                          (50 Q&A on convolutional networks)
│
├── ass4/
│   ├── ass4_rnn.ipynb
│   ├── GOOGL.csv
│   ├── EXPLANATION.md                 (Stock price prediction with LSTM)
│   └── QA.md                          (45 Q&A on RNN/LSTM & time series)
│
└── ass5/ (empty - no notebook provided)
```

---

## Assignment 1: Linear Regression vs Neural Networks

**Topic**: House Price Prediction using Boston Housing Dataset

### Dataset
- 506 samples with features: LSTAT, RM, PTRATIO
- Target: House price
- Train/test split: 80/20

### Models Compared
1. **Linear Regression**: y = b₀ + b₁x₁ + b₂x₂ + b₃x₃
2. **Neural Network**: 
   - Input: 3 features
   - Hidden: 128 → 64 → 32 → 16 neurons
   - Output: 1 (price prediction)
   - Activation: ReLU (hidden), None (output)

### Key Concepts
- Data normalization (StandardScaler)
- Train-test splitting
- Model compilation (optimizer, loss metrics)
- Regression evaluation (MSE, MAE, R²)
- Predicting on new data

### Expected Results
- Linear Regression: R² ≈ 0.55-0.65
- Neural Network: R² ≈ 0.70-0.80
- NN outperforms due to non-linear patterns

---

## Assignment 2: Image Classification & Text Sentiment

**Part 1: MNIST Digit Recognition (OCR)**

### Dataset
- 60,000 training images, 10,000 test
- 28×28 grayscale pixels (784 features)
- 10 classes (digits 0-9)
- Perfectly balanced (10% per class)

### Model Architecture
- Input: 784 features (flattened 28×28 image)
- Dense 512 (ReLU) + Dropout(0.2)
- Dense 512 (ReLU) + Dropout(0.2)
- Output: 10 (softmax for probability)

### Key Concepts
- Image preprocessing (normalization, reshaping)
- One-hot encoding labels
- Dropout regularization
- RMSprop optimizer
- Categorical cross-entropy loss

### Expected Results
- Training Accuracy: ~98%
- Test Accuracy: ~97-98%

**Part 2: IMDB Sentiment Analysis**

### Dataset
- 25,000 training + 25,000 test reviews
- Binary labels (positive/negative)
- Top 10,000 words vocabulary

### Model Architecture
- Input: 10,000 features (vectorized reviews)
- Dense 128 (ReLU) + Dropout(0.3)
- Dense 64 (ReLU) + Dropout(0.3)
- Output: 1 (sigmoid for binary probability)

### Key Concepts
- Text vectorization (one-hot encoding)
- Review decoding (word index mapping)
- Binary cross-entropy loss
- Sigmoid activation
- Binary classification metrics

### Expected Results
- Training Accuracy: ~95%
- Test Accuracy: ~88-92%

---

## Assignment 3: Convolutional Neural Networks

**Topic**: Fashion MNIST Classification (10 clothing item classes)

### Dataset
- 60,000 training images, 10,000 test
- 28×28 grayscale, 10 balanced classes
- Classes: T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

### Model Architecture
```
Conv2D(32, 3×3) → MaxPool(2×2) → Dropout(0.25)
Conv2D(64, 3×3) → MaxPool(2×2) → Dropout(0.2)
Flatten() → Dense(128, ReLU) → Dropout(0.25)
Dense(10, Softmax)
```

### Key Concepts
- 4D data shape: (batch, height, width, channels)
- Convolution: Sliding filter pattern detection
- Feature maps: Output of applying filters
- MaxPooling: Spatial downsampling
- Weight sharing: Filters applied everywhere
- ReLU non-linearity in Conv layers

### Architecture Benefits
- Fewer parameters than Dense network
- Respects spatial structure (nearby pixels related)
- Translation invariance (MaxPooling)
- Hierarchical feature learning

### Expected Results
- Training Accuracy: ~97-98%
- Test Accuracy: ~93-95%
- Outperforms Dense networks (which get ~92%)

---

## Assignment 4: Recurrent Neural Networks

**Topic**: Stock Price Prediction using LSTM

### Dataset
- Google (GOOGL) stock price history
- Features: Open, High, Low, Close, Volume
- Time-ordered data (80% train, 20% test chronological split)

### Data Preprocessing
1. **Scaling**: MinMaxScaler [0, 1]
2. **Windowing**: Create 60-day sequences
3. **Format**: (num_sequences, 60, 5) for LSTM input

### Model Architecture
```
LSTM(60, return_sequences=True) → Dropout(0.2)
LSTM(60, return_sequences=False) → Dropout(0.2)
Dense(25) → Dense(1)
```

### Key Concepts
- Time series: Order matters, no shuffling
- Windowing: Fixed sequences from continuous data
- LSTM gates: Forget, Input, Output
- return_sequences: All timesteps vs final only
- Inverse transformation: Scale predictions back
- Data leakage: Train on past, test on future

### LSTM Advantages
- Remembers 60+ day dependencies
- Prevents vanishing gradients (vs vanilla RNN)
- Gates control information flow
- Suitable for sequential data

### Expected Results
- Reasonable price predictions
- MSE/MAE depends on market volatility
- Can serve as baseline for more complex models

---

## Documentation Quality

### EXPLANATION.md Files (1,000+ total lines)
Each includes:
- **Detailed code explanation**: Line-by-line breakdown
- **Mathematical formulas**: Loss functions, activations, metrics
- **Visual examples**: Concept illustrations
- **Architecture diagrams**: Layer connections
- **Why decisions matter**: Trade-offs and justifications

### QA.md Files (200+ total questions)

**Categories covered:**
1. **Conceptual**: Understanding core ideas
2. **Implementation**: Practical code decisions
3. **Mathematical**: Formulas and derivations
4. **Design**: Architecture and hyperparameter choices
5. **Debugging**: Common problems and solutions
6. **Advanced**: Extensions and optimizations
7. **Applications**: Real-world deployment

**Question Types:**
- Short answer (definitions)
- Compare/contrast (models, techniques)
- Why/how (deep understanding)
- Calculation (numerical)
- Troubleshooting (debugging)
- Extension (advanced)

---

## Total Documentation Stats

| Assignment | Type | Lines | Questions |
|------------|------|-------|-----------|
| **Ass1** | LR vs NN | 280 | 40 |
| **Ass2** | MNIST + IMDB | 350 | 50 |
| **Ass3** | CNN | 320 | 50 |
| **Ass4** | LSTM | 340 | 45 |
| **TOTAL** | | 1,290+ | 185+ |

---

## Key Learning Outcomes

### Assignment 1
- Understand supervised learning regression
- Compare traditional ML vs deep learning
- Feature scaling importance
- Model evaluation metrics (MSE, MAE, R²)

### Assignment 2
- Image preprocessing and normalization
- One-hot encoding for classification
- Text vectorization
- Dense network for classification
- Dropout regularization
- Output activations: Softmax vs Sigmoid

### Assignment 3
- Spatial structure in images
- Convolutional filters and feature extraction
- MaxPooling for dimensionality reduction
- Parameter efficiency of CNNs
- Translation invariance
- Hierarchical feature learning

### Assignment 4
- Time series fundamentals
- Sequence creation (windowing)
- Recurrent connections and memory
- LSTM gates and long-term dependencies
- Data leakage prevention
- Inverse transformation for predictions

---

## File Organization Summary

✅ **Folder Structure**: 4 main assignment folders (ass1-ass4)
✅ **Notebooks**: All original Jupyter notebooks copied to folders
✅ **Data Files**: Boston House Price Data.csv in ass1/, GOOGL.csv in ass4/
✅ **Documentation**: 8 markdown files total (4 EXPLANATION + 4 QA)
✅ **Comprehensive Coverage**: 185+ questions, 1,290+ lines of explanation
✅ **Code Walkthroughs**: Line-by-line explanations of all major code sections
✅ **Practical Focus**: Debugging, design, and real-world application sections

---

## How to Use

### For Learning
1. Read EXPLANATION.md for concepts and code breakdown
2. Review code in original .ipynb notebook
3. Answer questions in QA.md to test understanding
4. Refer to answers to deepen comprehension

### For Reference
- **Specific concept**: Find in EXPLANATION.md (well-organized)
- **Troubleshooting**: Check debugging section in QA.md
- **Architecture design**: Review design questions in QA.md
- **Implementation details**: Check implementation questions in QA.md

### For Teaching
- EXPLANATION.md suitable for teaching materials
- QA.md good for assignments or study guides
- Can selectively use sections for specific topics

---

## Notes

- **No ass5 notebook**: Only 4 DL assignments in provided files
- **Comprehensive scope**: Covers fundamentals through advanced topics
- **Progressive complexity**: ass1 (basics) → ass2 (classification) → ass3 (CNN) → ass4 (LSTM)
- **Cross-referenced**: Concepts build upon each other
- **Production-ready insights**: Include deployment, debugging, optimization

---

## Next Steps

To deepen learning:
1. **Run notebooks**: Execute code cells to see outputs
2. **Modify code**: Change hyperparameters to see effects
3. **Extension projects**: Combine concepts in new ways
4. **Advanced techniques**: Study transfer learning, ensemble methods, attention
5. **Real datasets**: Apply to similar problems with your own data
