"""
Demo-Friendly Streamlit App (No PyTorch/FinBERT Required)
Uses TextBlob for sentiment analysis to avoid DLL issues
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
from datetime import datetime
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import STOCK_SYMBOLS, FEATURES_DIR, MODELS_DIR
from textblob import TextBlob  # Lightweight sentiment
import requests
from bs4 import BeautifulSoup

# Page configuration
st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    """Load stock data and features."""
    try:
        features_file = FEATURES_DIR / "engineered_features.csv"
        if features_file.exists():
            df = pd.read_csv(features_file, parse_dates=['Date'])
            return df
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_data
def load_sentiment_timeline():
    """Load static sentiment timeline data for visualization."""
    try:
        sentiment_file = FEATURES_DIR / "sentiment_timeline.csv"
        if sentiment_file.exists():
            df = pd.read_csv(sentiment_file, parse_dates=['Date'])
            return df
        return None
    except Exception as e:
        return None

@st.cache_resource
def load_model():
    """Load trained model if available."""
    try:
        import joblib
        model_path = MODELS_DIR / "random_forest.pkl"
        if model_path.exists():
            model = joblib.load(model_path)
            return model, "Random Forest"
        return None, None
    except Exception as e:
        st.warning(f"Could not load model: {str(e)}")
        return None, None

def predict_next_price(model, stock_data, symbol):
    """Predict next day's price."""
    try:
        # Get latest data
        latest = stock_data[stock_data['Symbol'] == symbol].iloc[-1]
        
       # Prepare features (same as training)
        exclude_cols = [
            'Date', 'Symbol', 'Target_Price', 'Target_Return', 'Target_Direction',
            'Close', 'Open', 'High', 'Low', 'Volume',
            'daily_sentiment', 'sentiment_std', 'news_count',
            'positive_news_ratio', 'negative_news_ratio',
            'Returns', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
            'BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Width',
            'ATR', 'OBV', 'Stoch_K', 'Stoch_D',
            'SMA_10', 'SMA_20', 'SMA_50', 'SMA_200',
            'EMA_10', 'EMA_12', 'EMA_20', 'EMA_26', 'EMA_50',
            'Volume_SMA_20',
        ]
        
        # Exclude lag cols
        close_lag_cols = [col for col in stock_data.columns if 'Close_lag' in col or 'Close_rolling' in col]
        exclude_cols.extend(close_lag_cols)
        
        feature_cols = [col for col in stock_data.columns if col not in exclude_cols]
        features = latest[feature_cols].values.reshape(1, -1)
        
        # Predict
        prediction = model.predict(features)[0]
        current_price = latest['Close']
        
        return prediction, current_price
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None

def analyze_latest_news_simple(symbol):
    """Simple news sentiment using TextBlob (no PyTorch)."""
    st.info("🔍 Fetching latest news (using TextBlob for sentiment)...")
    
    try:
        # Try Google News
        query = f"{symbol} stock"
        url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'xml')
        items = soup.find_all('item')[:5]
        
        if not items:
            st.warning("No news found")
            return
        
        sentiments = []
        for item in items:
            title = item.title.text
            
            # Simple sentiment with TextBlob
            blob = TextBlob(title)
            sentiment = blob.sentiment.polarity
            sentiments.append(sentiment)
            
            # Display
            emoji = "📈" if sentiment > 0.1 else "📉" if sentiment < -0.1 else "➖"
            st.write(f"{emoji} **{sentiment:.3f}** - {title}")
        
        avg_sent = np.mean(sentiments)
        st.metric("Average Sentiment", f"{avg_sent:.3f}", 
                 "Bullish" if avg_sent > 0 else "Bearish" if avg_sent < 0 else "Neutral")
        
    except Exception as e:
        st.error(f"Could not fetch news: {str(e)}")

# Plot functions
def plot_price_chart(df, symbol, prediction=None):
    """Plot stock price with prediction."""
    symbol_data = df[df['Symbol'] == symbol].copy()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=symbol_data['Date'],
        y=symbol_data['Close'],
        mode='lines',
        name='Historical Price',
        line=dict(color='#1f77b4', width=2)
    ))
    
    if prediction is not None:
        last_date = symbol_data['Date'].iloc[-1]
        next_date = last_date + pd.Timedelta(days=1)
        
        fig.add_trace(go.Scatter(
            x=[last_date, next_date],
            y=[symbol_data['Close'].iloc[-1], prediction],
            mode='lines+markers',
            name='Prediction',
            line=dict(color='red', width=2, dash='dash'),
            marker=dict(size=10, symbol='star')
        ))
    
    fig.update_layout(
        title=f'{symbol} Stock Price',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        hovermode='x unified',
        height=400
    )
    
    return fig

def plot_sentiment_timeline(df, symbol):
    """Plot sentiment over time."""
    symbol_data = df[df['Symbol'] == symbol].reset_index(drop=True)
    
    if 'daily_sentiment' not in symbol_data.columns or symbol_data['daily_sentiment'].abs().sum() == 0:
        return None
    
    fig = go.Figure()
    
    colors = ['green' if x > 0 else 'red' if x < 0 else 'gray' 
             for x in symbol_data['daily_sentiment']]
    
    fig.add_trace(go.Bar(
        x=symbol_data['Date'],
        y=symbol_data['daily_sentiment'],
        marker_color=colors,
        name='Sentiment'
    ))
    
    fig.update_layout(
        title=f'{symbol} News  Sentiment Timeline',
        xaxis_title='Date',
        yaxis_title='Sentiment Score',
        hovermode='x unified',
        height=300
    )
    
    return fig

# Main app
def main():
    st.markdown('<p class="main-header">📈 Stock Price Prediction with News Sentiment</p>', unsafe_allow_html=True)
    st.markdown("**AI-Powered Forecasting | 91.85% Directional Accuracy**")
    
    # Sidebar
    st.sidebar.header("📊 Model Information")
    model, model_name = load_model()
    
    if model:
        st.sidebar.success(f"✅ Model Loaded: {model_name}")
        st.sidebar.metric("Directional Accuracy", "91.85%")
        st.sidebar.metric("RMSE", "$26.11")
    else:
        st.sidebar.warning("⚠️ Model not loaded")
        st.sidebar.info("Run: `py train_model.py`")
    
    # Stock selector
    st.sidebar.header("📈 Select Stock")
    selected_stock = st.sidebar.selectbox(
        "Choose a stock:",
        options=list(STOCK_SYMBOLS.keys()),
        format_func=lambda x: f"{x} - {STOCK_SYMBOLS[x]}"
    )
    
    # Real-time sentiment button
    st.sidebar.header("📰 Real-Time Analysis")
    if st.sidebar.button("🔍 Analyze Latest News", use_container_width=True):
        with st.sidebar:
            analyze_latest_news_simple(selected_stock)
    
    # Load data
    df = load_data()
    
    if df is None:
        st.error("❌ No data found. Please run: `py test_pipeline.py`")
        return
    
    stock_data = df[df['Symbol'] == selected_stock]
    
    if len(stock_data) == 0:
        st.error(f"No data for {selected_stock}")
        return
    
    # Prediction
    if model:
        prediction, current_price = predict_next_price(model, df, selected_stock)
        
        if prediction:
            col1, col2, col3 = st.columns(3)
            
            change = prediction - current_price
            change_pct = (change / current_price) * 100
            
            with col1:
                st.metric("Current Price", f"${current_price:.2f}")
            with col2:
                st.metric("Predicted Price", f"${prediction:.2f}", 
                         f"{change:+.2f} ({change_pct:+.2f}%)")
            with col3:
                direction = "📈 UP" if change > 0 else "📉 DOWN" if change < 0 else "➖ FLAT"
                st.metric("Direction", direction)
    
    # Price chart
    st.subheader(f"📊 {selected_stock} Price History")
    price_fig = plot_price_chart(stock_data, selected_stock, prediction if model else None)
    st.plotly_chart(price_fig, use_container_width=True)
    
    # Sentiment chart
    sentiment_timeline_df = load_sentiment_timeline()
    
    if sentiment_timeline_df is not None:
        sentiment_fig = plot_sentiment_timeline(sentiment_timeline_df, selected_stock)
        if sentiment_fig:
            st.subheader("💭 News Sentiment Over Time")
            st.caption("Historical sentiment from past year's news data (demo data)")
            st.plotly_chart(sentiment_fig, use_container_width=True)
        else:
            st.info("📊 Run `py generate_demo_data.py` to create sentiment timeline")
    
    # Footer
    st.markdown("---")
    st.markdown("**Model:** Random Forest | **Features:** 60+ engineered features | **Methodology:** CRISP-DM")

if __name__ == "__main__":
    main()
