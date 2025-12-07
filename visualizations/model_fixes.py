import yfinance as yf
import pandas as pd
import pickle

from sklearn.ensemble import RandomForestRegressor

# --- Load existing model ---
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# --- Get new stock data ---
df = yf.download("AAPL", period="3mo")
df["return"] = df["Close"].pct_change()
df = df.dropna()

X_new = df[["Open", "High", "Low", "Close", "Volume"]]
y_new = df["return"]

# --- Retrain (fix) the model ---
model.fit(X_new, y_new)

# --- Save updated model ---
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model updated and fixed successfully.")
