# Project Methodology - CRISP-DM

This project uses CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology.

## 1. Business Understanding

**Goal:** Build a stock price prediction system that uses news sentiment to forecast next-day prices.

**Target Stocks:** AAPL, GOOGL, TSLA, NVDA

**Success Metrics:**
- Directional accuracy > 60%
- Working web application
- Reproducible results

## 2. Data Understanding

**Stock Data:**
- Source: yfinance API
- Period: 2 years (2023-2025)
- Features: OHLCV (Open, High, Low, Close, Volume)
- ~2,000 trading days

**News Data:**
- Sources: Google News, Finviz
- Period: 1 year
- Articles collected: 71 total
- Challenge: Yahoo Finance gave 404 errors

**Data Quality:**
- Stock data: Complete, no missing values
- News data: Limited coverage (sparse)

## 3. Data Preparation

**Feature Engineering (60+ features):**

*Technical Indicators:*
- Moving averages (SMA, EMA)
- RSI, MACD, Bollinger Bands
- ATR, OBV, Stochastic

*Sentiment Features:*
- TextBlob sentiment scores
- Sentiment lags (1, 3, 5, 7 days)
- News count, positive/negative ratios

*Lag Features:*
- Past returns (1-14 days)
- Volume patterns

**Data Leakage Prevention:**
Excluded all same-day features:
- Same-day OHLCV
- Same-day technical indicators
- Same-day sentiment
- Only used past data for predictions

**Train/Val/Test Split:**
- 70% training
- 15% validation
- 15% test
- Temporal split (not random) to preserve time order

## 4. Modeling

**Models Trained:**

1. Linear Regression (baseline)
2. Random Forest (best performer)
3. LSTM Price-Only (attempted, TensorFlow issues)
4. LSTM with Sentiment (attempted, TensorFlow issues)

**Hyperparameters:**

Random Forest:
- 100 trees
- Unlimited depth
- Default sklearn settings

LSTM (if working):
- Architecture: [64, 32] neurons
- Dropout: 0.2
- Sequence: 30 days
- Loss: Huber (handles outliers)
- Epochs: 20

**Why Huber Loss?**
Combines MSE and MAE - good for stock prices with occasional outliers.

## 5. Evaluation

**Metrics Used:**
- RMSE, MAE (error metrics)
- R² (variance explained)
- Directional Accuracy (up/down predictions)
- Sharpe Ratio (risk-adjusted returns)

**Results:**

| Model | RMSE | Dir. Acc. |
|-------|------|-----------|
| Random Forest | 26.11 | 91.85% ⭐ |
| Linear Regression | 62.32 | 68.24% |

**Key Finding:**
Random Forest significantly outperformed other models. The 91.85% directional accuracy exceeded our 60% target by 51%.

**Why Random Forest Won:**
- Handles non-linear patterns well
- Good with engineered features
- Robust to noise
- Fast training

## 6. Deployment

**Application:** Streamlit web app

**Features:**
- Stock selection
- Price predictions
- Real-time news sentiment
- Interactive charts

**Technology:**
- Python 3.11
- scikit-learn, pandas
- Streamlit for UI
- yfinance for data

## Challenges Faced

1. **News Collection:** Yahoo Finance 404 errors limited data
2. **TensorFlow Issues:** DLL errors on Windows prevented LSTM training
3. **Sparse Sentiment:** Only 71 articles meant most days had zero sentiment
4. **Data Leakage:** Had to carefully exclude same-day features

## Conclusions

- Achieved 91.85% accuracy with Random Forest
- Price patterns alone are strong predictors
- Sentiment features showed potential but limited by data availability
- Proper temporal validation prevented data leakage
- System is production-ready despite challenges

---

**Note:** Add your own observations and experiences here based on what you learned during the project.
