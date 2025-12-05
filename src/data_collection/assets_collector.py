import yfinance as yf

def get_stock_data(ticker, period="1mo", interval="1d"):
    """
    Collect stock price data using yfinance.

    Args:
        ticker (str): Stock ticker symbol, e.g., "AAPL"
        period (str): How far back to retrieve data.
                      Examples: "1d", "5d", "1mo", "6mo", "1y", "5y", "max"
        interval (str): Data frequency.
                        Examples: "1m", "5m", "1h", "1d", "1wk", "1mo"

    Returns:
        pandas.DataFrame: Historical stock price data
    """
    stock = yf.Ticker(ticker)
    data = stock.history(period=period, interval=interval)
    return data
