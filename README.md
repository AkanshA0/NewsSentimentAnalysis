# Stock Price Prediction with News Sentiment Analysis

A comprehensive data mining project implementing stock price prediction using ensemble models with news sentiment analysis, following CRISP-DM methodology.

## 🎯 Project Overview

This project predicts stock prices for **AAPL, GOOGL, TSLA, and NVDA** using:
- **Historical price data** (2 years from yfinance)
- **News sentiment analysis** (FinBERT + TextBlob)
- **Ensemble modeling** (Baseline + Price LSTM + Sentiment LSTM + Multi-Input LSTM)

### Key Features
- ✅ **100% Free** - No API keys required
- ✅ **CRISP-DM Methodology** - Complete data mining lifecycle
- ✅ **Ensemble Models** - Multiple models for robust predictions
- ✅ **Ablation Studies** - Comprehensive model comparison
- ✅ **20+ Visualizations** - Extensive model evaluation
- ✅ **MLOps Pipeline** - Airflow + MLflow integration
- ✅ **Production Ready** - Streamlit web application

## 📊 Data Sources

### Stock Data
- **Source**: yfinance (free, no API key)
- **Period**: 2 years of daily data
- **Features**: OHLCV + 15 technical indicators (RSI, MACD, Bollinger Bands, etc.)

### News Data
- **Yahoo Finance**: Web scraping for news articles
- **Google News**: RSS feeds for real-time news
- **Finviz**: Financial news scraping
- **Sentiment**: FinBERT (financial-specific) + TextBlob (backup)

## 🏗️ Project Structure

```
NewsSentiment/
├── data/
│   ├── raw/                    # Raw stock and news data
│   ├── processed/              # Cleaned data
│   └── features/               # Engineered features
├── src/
│   ├── data_collection/        # Stock & news collectors
│   ├── preprocessing/          # Data cleaning & feature engineering
│   ├── models/                 # Model implementations
│   ├── evaluation/             # Metrics & visualizations
│   └── utils/                  # Configuration & utilities
├── app/                        # Streamlit web application
├── airflow/                    # Airflow DAGs
├── mlflow/                     # MLflow tracking
├── notebooks/                  # Jupyter/Colab notebooks
├── models/                     # Saved models
├── visualizations/             # Generated plots
└── requirements.txt
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd NewsSentiment

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Collection

```bash
# Collect stock data
python src/data_collection/stock_collector.py

# Collect news data
python src/data_collection/news_collector.py
```

### 3. Data Preprocessing

```bash
# Clean data
python src/preprocessing/data_cleaner.py

# Engineer features (includes sentiment analysis)
python src/preprocessing/feature_engineer.py
```

### 4. Model Training

```bash
# Train all models
python src/models/train_ensemble.py
```

### 5. Run Web Application

```bash
streamlit run app/app.py
```

## 🤖 Model Architecture

### Ensemble Approach

We train 4 models and combine their predictions:

1. **Baseline Models** (ARIMA, Linear Regression)
   - Simple benchmarks for comparison
   
2. **Price-Only LSTM**
   - Input: Historical prices + technical indicators
   - Architecture: Bidirectional LSTM layers
   
3. **Sentiment-Enhanced LSTM**
   - Input: Prices + technical indicators + news sentiment
   - Architecture: LSTM with sentiment features
   
4. **Multi-Input LSTM**
   - Separate branches for price and sentiment
   - Merged layers for final prediction

### Final Prediction

```python
final_prediction = (
    0.1 * baseline +
    0.3 * price_lstm +
    0.3 * sentiment_lstm +
    0.3 * multi_input_lstm
)
```

## 📈 Evaluation Metrics

- **Regression**: RMSE, MAE, MAPE, R²
- **Classification**: Directional Accuracy (up/down)
- **Financial**: Sharpe Ratio, Maximum Drawdown
- **20+ Visualizations**: Learning curves, confusion matrices, feature importance, etc.

## 🔄 MLOps Pipeline

### Airflow DAGs
- **Daily**: Data collection and prediction generation
- **Weekly**: Model retraining and evaluation

### MLflow Tracking
- All experiments logged with parameters, metrics, and artifacts
- Model registry for version control
- Automatic model promotion based on performance

## 💻 Web Application

Interactive Streamlit app with:
- Stock selector (AAPL, GOOGL, TSLA, NVDA)
- Real-time predictions
- Interactive visualizations
- Model comparison
- Retraining interface

## 📚 Documentation

- **Implementation Plan**: See `implementation_plan.md`
- **Task Breakdown**: See `task.md`
- **Colab Notebook**: Heavily documented with parameter justifications
- **Walkthrough**: Complete execution summary with results

## 🎓 Academic Requirements

✅ **CRISP-DM Methodology**: Complete lifecycle implementation
✅ **Original Code**: Written from scratch (Turnitin-safe)
✅ **Heavy Documentation**: All parameters and choices explained
✅ **20% Visualizations**: Comprehensive evaluation dashboard
✅ **Production Demo**: Fully functional web application
✅ **Ablation Studies**: Model component analysis
✅ **MLOps Bonus**: Airflow + MLflow integration

## 📝 License

This project is for academic purposes.

## 👥 Author

[Your Name]

## 🙏 Acknowledgments

- FinBERT model by ProsusAI
- yfinance for free stock data
- Open-source community
