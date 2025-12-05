# Stock Price Prediction with News Sentiment

Predicting next-day stock prices using machine learning and news sentiment analysis.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Collect data
python test_pipeline.py

# Train models
python train_baseline_only.py

# Run app
streamlit run app\app.py
```

## Project Overview

This system predicts stock prices for AAPL, GOOGL, TSLA, and NVDA by combining:
- Historical price data (2 years)
- Technical indicators (RSI, MACD, etc.)
- News sentiment analysis
- Machine learning models

**Best Result:** 91.85% directional accuracy with Random Forest

## Features

- 60+ engineered features
- Multiple ML models (Linear Regression, Random Forest)
- Real-time news sentiment
- Interactive web dashboard
- No data leakage (proper temporal validation)

## Tech Stack

- Python 3.11
- scikit-learn (ML models)
- pandas (data processing)
- Streamlit (web app)
- yfinance (stock data)
- TextBlob (sentiment analysis)

## Project Structure

```
NewsSentiment/
├── app/                    # Streamlit web app
├── data/                   # Data files
├── models/                 # Trained models
├── src/                    # Source code
│   ├── data_collection/   # Stock & news collectors
│   ├── preprocessing/     # Feature engineering
│   ├── models/            # ML models
│   └── evaluation/        # Metrics & visualizations
├── train_baseline_only.py # Training script
└── test_pipeline.py       # Data pipeline
```

## How It Works

1. **Data Collection:** Fetch stock prices from yfinance, scrape news from Google News/Finviz
2. **Feature Engineering:** Create 60+ features including technical indicators and sentiment scores
3. **Model Training:** Train Random Forest and Linear Regression models
4. **Prediction:** Use trained models to predict next-day prices
5. **Deployment:** Streamlit app for interactive predictions

## Results

| Model | RMSE | MAE | Directional Accuracy |
|-------|------|-----|---------------------|
| Random Forest | 26.11 | 16.29 | 91.85% |
| Linear Regression | 62.32 | 37.37 | 68.24% |

## Methodology

Follows CRISP-DM framework:
1. Business Understanding - Define prediction goals
2. Data Understanding - Analyze stock and news data
3. Data Preparation - Engineer features, prevent data leakage
4. Modeling - Train and compare models
5. Evaluation - Test on unseen data
6. Deployment - Web application

See `CRISP_DM_METHODOLOGY.md` for details.

## Data Leakage Prevention

Critical: Only use PAST data for predictions!

**Excluded features:**
- Same-day OHLCV (not known until market closes)
- Same-day technical indicators
- Same-day news sentiment
- Target variable lags

**Used features:**
- Past returns (1-14 days ago)
- Lagged sentiment
- Volume patterns
- Rolling statistics

## Challenges

1. **Limited News Data:** Only collected 71 articles due to API limitations
2. **TensorFlow Issues:** DLL errors prevented LSTM training on Windows
3. **Sparse Sentiment:** Most days have zero news coverage

Despite these challenges, achieved excellent results with Random Forest!

## Future Improvements

- Fix news collection (more sources, paid APIs)
- Add more stocks
- Implement LSTM models (fix TensorFlow)
- Cloud deployment
- Automated retraining

## Academic Compliance

- Original code (no plagiarism)
- Proper citations for libraries
- No data leakage
- Reproducible results
- Complete documentation

## License

MIT License

## Contact

[Add your contact info]

---

**Note:** This is an academic project. Not financial advice.
