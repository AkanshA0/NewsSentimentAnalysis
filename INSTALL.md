# Simple Installation for Python 3.11

## Quick Install (No pandas-ta needed!)

```bash
# Install all required packages
pip install numpy pandas scikit-learn scipy
pip install tensorflow torch transformers  
pip install yfinance beautifulsoup4 lxml requests feedparser
pip install nltk textblob
pip install matplotlib seaborn plotly
pip install streamlit
pip install statsmodels python-dotenv tqdm joblib
```

## That's it! 

The project now calculates technical indicators manually, so you don't need pandas-ta.

## Test Installation

```bash
python test_pipeline.py
```

This will:
1. ✅ Collect stock data
2. ✅ Calculate technical indicators (manually)
3. ✅ Collect news
4. ✅ Analyze sentiment
5. ✅ Create features

## If You Get Errors

### TensorFlow issues:
```bash
pip install tensorflow-cpu
```

### PyTorch issues:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Minimal test (just data collection):
```bash
pip install pandas numpy yfinance beautifulsoup4 requests
python -c "from src.data_collection.stock_collector import StockDataCollector; print('Works!')"
```

That's all you need!
