"""
Enhanced Streamlit App with Predictions and Real-Time Sentiment
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import sys
from datetime import datetime
import joblib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import STOCK_SYMBOLS, FEATURES_DIR, MODELS_DIR
from src.data_collection.news_collector import NewsCollector
from src.preprocessing.feature_engineer import FeatureEngineer

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
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load stock data and features."""
    try:
        features_file = FEATURES_DIR / "engineered_features.csv"
        if features_file.exists():
            df = pd.read_csv(features_file, parse_dates=['Date'])
            
            # Check if data is fresh (less than 1 day old)
            last_date = df['Date'].max()
            # Make both timestamps timezone-naive for comparison
            now = pd.Timestamp.now().tz_localize(None)
            last_date = pd.to_datetime(last_date).tz_localize(None)
            days_old = (now - last_date).days
            
            if days_old > 1:
                st.warning(f"⚠️ Data is {days_old} days old (last update: {last_date.date()})")
                st.info("💡 For latest predictions, click 'Refresh Data' in sidebar or run: `python test_pipeline.py`")
            
            return df
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def refresh_data():
    """Refresh stock data by running data collection."""
    try:
        import subprocess
        st.info("🔄 Fetching latest stock data...")
        
        # Run data collection
        result = subprocess.run(
            ["python", "test_pipeline.py"],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            st.success("✅ Data refreshed successfully!")
            st.cache_data.clear()  # Clear cache to reload new data
            st.rerun()
        else:
            st.error(f"❌ Data refresh failed: {result.stderr}")
    except Exception as e:
        st.error(f"Error refreshing data: {str(e)}")


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
        # Try Random Forest first (no TensorFlow needed)
        rf_model_path = MODELS_DIR / "random_forest.pkl"
        scaler_path = MODELS_DIR / "scaler_baseline.pkl"
        feature_cols_path = MODELS_DIR / "feature_cols.pkl"
        
        if rf_model_path.exists() and scaler_path.exists() and feature_cols_path.exists():
            import joblib
            model = joblib.load(rf_model_path)
            scaler = joblib.load(scaler_path)
            feature_cols = joblib.load(feature_cols_path)
            return model, scaler, feature_cols
        
        # Fallback to LSTM if Random Forest not found
        lstm_model_path = MODELS_DIR / "lstm_model.h5"
        lstm_scaler_path = MODELS_DIR / "scaler.pkl"
        
        if lstm_model_path.exists() and lstm_scaler_path.exists():
            from tensorflow import keras
            import joblib
            model = keras.models.load_model(lstm_model_path)
            scaler = joblib.load(lstm_scaler_path)
            feature_cols = joblib.load(MODELS_DIR / "feature_cols.pkl")
            return model, scaler, feature_cols
            
        return None, None, None
    except Exception as e:
        st.warning(f"Could not load model: {str(e)}")
        return None, None, None


def get_realtime_sentiment(symbol):
    """Get real-time news sentiment AND latest stock price for today."""
    try:
        # Fetch latest stock price
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period='1d')
        
        latest_price = None
        if not hist.empty:
            latest_price = hist['Close'].iloc[-1]
        
        # Fetch news
        collector = NewsCollector(symbols=[symbol])
        news_df = collector.collect_all_news(max_articles_per_source=100)
        
        if len(news_df) > 0:
            engineer = FeatureEngineer(use_finbert=False)  # Use TextBlob for speed
            news_with_sentiment = engineer.add_sentiment_features(news_df)
            
            avg_sentiment = news_with_sentiment['sentiment_score'].mean()
            news_count = len(news_with_sentiment)
            
            return avg_sentiment, news_count, news_with_sentiment, latest_price
        return 0.0, 0, pd.DataFrame(), latest_price
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return 0.0, 0, pd.DataFrame(), None


def plot_price_with_prediction(df, symbol, model, scaler, feature_cols):
    """Create price chart with predictions."""
    symbol_data = df[df['Symbol'] == symbol].copy().sort_values('Date')
    
    fig = go.Figure()
    
    # Historical prices (BLUE LINE)
    fig.add_trace(go.Scatter(
        x=symbol_data['Date'],
        y=symbol_data['Close'],
        name='Historical Price',
        line=dict(color='#1f77b4', width=2)  # Blue
    ))
    
    # Add predictions if model is available
    if model is not None:
        try:
            # Check if model is LSTM (needs sequences) or Random Forest (single row)
            model_type = str(type(model).__name__)
            
            if 'RandomForest' in model_type or 'Linear' in model_type:
                # Random Forest or Linear Regression - use latest single row
                recent_data = symbol_data[feature_cols].tail(1).values
                recent_scaled = scaler.transform(recent_data)
                prediction = model.predict(recent_scaled)[0]  # No verbose parameter
            else:
                # LSTM - use sequence of 30 days
                recent_data = symbol_data[feature_cols].tail(30).values
                recent_scaled = scaler.transform(recent_data)
                recent_seq = recent_scaled.reshape(1, 30, -1)
                prediction = model.predict(recent_seq, verbose=0)[0][0]  # verbose only for LSTM
            
            # Add prediction point (RED STAR)
            last_date = symbol_data['Date'].iloc[-1]
            next_date = last_date + pd.Timedelta(days=1)
            
            fig.add_trace(go.Scatter(
                x=[next_date],
                y=[prediction],
                name='Next Day Prediction',
                mode='markers',
                marker=dict(color='red', size=20, symbol='star',
                           line=dict(color='darkred', width=2))  # Red star
            ))
            
            # Add dashed line from last price to prediction (RED DASHED)
            fig.add_trace(go.Scatter(
                x=[last_date, next_date],
                y=[symbol_data['Close'].iloc[-1], prediction],
                name='Prediction Trend',
                mode='lines',
                line=dict(color='red', width=2, dash='dash'),
                showlegend=False
            ))
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            pass
    
    fig.update_layout(
        title=f'{symbol} Stock Price with Prediction',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    return fig


def plot_sentiment_timeline(df, symbol):
    """Create sentiment timeline."""
    symbol_data = df[df['Symbol'] == symbol].copy()
    
    if 'daily_sentiment' not in symbol_data.columns:
        return None
    
    fig = go.Figure()
    
    # Filter out days with 0 sentiment to show only relevant data
    non_zero_sentiment = symbol_data[symbol_data['daily_sentiment'] != 0]
    
    if len(non_zero_sentiment) > 0:
        plot_data = non_zero_sentiment
    else:
        # Fallback if all are zero (shouldn't happen with fresh data)
        plot_data = symbol_data.tail(30)
    
    colors = ['green' if x > 0 else 'red' for x in plot_data['daily_sentiment']]
    
    fig.add_trace(go.Bar(
        x=plot_data['Date'],
        y=plot_data['daily_sentiment'],
        name='Daily Sentiment',
        marker_color=colors
    ))
    
    fig.update_layout(
        title=f'{symbol} News Sentiment Over Time',
        xaxis_title='Date',
        yaxis_title='Sentiment Score',
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    return fig


def main():
    """Main application."""
    
    # Header
    st.markdown('<h1 class="main-header">AI Stock Price Predictor</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <p style='text-align: center; font-size: 1.2rem; color: #666;'>
    AI-Powered predictions using Random Forest + News Sentiment Analysis
    </p>
    """, unsafe_allow_html=True)
    
    # Load model
    model, scaler, feature_cols = load_model()
    model_trained = model is not None
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Stock Selector")
        
        selected_stock = st.selectbox(
            "Choose Stock",
            STOCK_SYMBOLS,
            help="Select a stock to analyze"
        )
        
        st.markdown("---")
        
        # Data refresh button
        st.subheader("🔄 Data Management")
        if st.button("Refresh Data", help="Fetch latest stock prices and news"):
            refresh_data()
        
        st.markdown("---")
        
        # Real-time sentiment
        st.subheader("📰 Today's News Sentiment")
        if st.button("🔍 Analyze Latest News"):
            with st.spinner("Fetching latest news and stock price..."):
                sentiment, count, news_df, latest_price = get_realtime_sentiment(selected_stock)
                
                if count > 0:
                    st.success(f"✅ Analyzed {count} articles")
                    
                    # Show latest price if available
                    if latest_price:
                        st.metric("📈 Latest Stock Price", f"${latest_price:.2f}", 
                                 "Live from yfinance", delta_color="off")
                    
                    st.metric("Today's Sentiment", f"{sentiment:.3f}",
                             delta="Bullish" if sentiment > 0 else "Bearish")
                    
                    with st.expander("📄 Latest Headlines"):
                        for _, row in news_df.head(5).iterrows():
                            sentiment_emoji = "🟢" if row['sentiment_score'] > 0 else "🔴"
                            st.write(f"{sentiment_emoji} {row['title'][:80]}...")
                else:
                    st.warning("No recent news found")
        
        st.markdown("---")
    
    # Load data
    df = load_data()
    
    if df is None:
        st.error("⚠️ No data available. Run `python test_pipeline.py` first.")
        return
    
    stock_data = df[df['Symbol'] == selected_stock].copy()
    
    if len(stock_data) == 0:
        st.warning(f"No data for {selected_stock}")
        return
    
    stock_data = stock_data.sort_values('Date')  # Ensure sorted by date
    
    # Prediction Box
    if model_trained:
        try:
            # Check model type
            model_type = str(type(model).__name__)
            
            if 'RandomForest' in model_type or 'Linear' in model_type:
                # Random Forest - use single row
                recent_data = stock_data[feature_cols].tail(1).values
                recent_scaled = scaler.transform(recent_data)
                prediction = model.predict(recent_scaled)[0]
            else:
                # LSTM - use sequence
                recent_data = stock_data[feature_cols].tail(30).values
                recent_scaled = scaler.transform(recent_data)
                recent_seq = recent_scaled.reshape(1, 30, -1)
                prediction = model.predict(recent_seq, verbose=0)[0][0]
            
            current_price = stock_data['Close'].iloc[-1]
            predicted_change = ((prediction - current_price) / current_price) * 100
            
            st.markdown(f"""
            <div class="prediction-box">
                <h2>🎯 Next Day Prediction for {selected_stock}</h2>
                <h1 style='font-size: 3rem; margin: 1rem 0;'>${prediction:.2f}</h1>
                <p style='font-size: 1.5rem;'>
                    {'📈 UP' if predicted_change > 0 else '📉 DOWN'} {abs(predicted_change):.2f}%
                </p>
                <p style='opacity: 0.8;'>Based on last historical price: ${current_price:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
    else:
        st.info("📊 Prediction models are being prepared. Historical data and analysis available below.")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    current_price = stock_data['Close'].iloc[-1]
    price_change_pct = ((stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[-2]) / 
                       stock_data['Close'].iloc[-2]) * 100 if len(stock_data) > 1 else 0
    
    with col1:
        st.metric("Last Historical Price", f"${current_price:.2f}", f"{price_change_pct:+.2f}%",
                 help="Last closing price from training dataset")
    
    with col2:
        # Use latest sentiment (most recent date)
        latest_sentiment = stock_data['daily_sentiment'].iloc[-1] if 'daily_sentiment' in stock_data.columns else 0
        sentiment_date = stock_data['Date'].iloc[-1].strftime('%Y-%m-%d')
        
        # Fix -0.000 display issue
        if abs(latest_sentiment) < 0.001:
            latest_sentiment = 0.0
            
        # Use numeric value for delta to get correct Green/Red arrow colors
        st.metric("Latest Sentiment", f"{latest_sentiment:.3f}", f"{latest_sentiment:.3f}",
                 help=f"Sentiment score for {sentiment_date}")
    
    with col3:
        st.metric("Data Points", f"{len(stock_data)}", "days",
                 help="Number of trading days in dataset")
    
    with col4:
        news_count = stock_data['news_count'].sum() if 'news_count' in stock_data.columns else 0
        st.metric("News Analyzed", f"{int(news_count)}", "articles",
                 help="Total news articles in training data")
    
    # Charts
    st.markdown("---")
    
    # Price chart with predictions
    st.subheader(f"📊 {selected_stock} Price Analysis")
    price_fig = plot_price_with_prediction(df, selected_stock, model, scaler, feature_cols)
    st.plotly_chart(price_fig, use_container_width=True)
    
    # Sentiment chart
    if 'daily_sentiment' in stock_data.columns and stock_data['daily_sentiment'].abs().sum() > 0:
        st.subheader(f"💭 Sentiment Timeline")
        sentiment_fig = plot_sentiment_timeline(df, selected_stock)
        if sentiment_fig:
            st.plotly_chart(sentiment_fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <p style='text-align: center; color: #666;'>
    🚀 Built with Random Forest, TextBlob, and Streamlit | Data: yfinance, Google News, Finviz
    </p>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
