from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import matplotlib.pyplot as plt
import time
import streamlit as st
from sqlalchemy import create_engine

# MySQL Connection Details (Environment Variables for Docker/Kubernetes)
MYSQL_USER = "root"
MYSQL_PASSWORD = "password"
MYSQL_HOST = "db"  # Docker Compose service name for MySQL
MYSQL_PORT = 3306
MYSQL_DB = "stocks"
DATABASE_URL = f"mysql+mysqlconnector://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"

# Set up SQLAlchemy Engine
engine = create_engine(DATABASE_URL)

# Alpha Vantage API Key
api_key = "GH9ZJMSFZ3HP1RXA"
ts = TimeSeries(key=api_key, output_format="pandas")

# Helper Functions
def calculate_rsi(data, window=14):
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    macd_line = data['Close'].ewm(span=fast_period, min_periods=1).mean() - \
                data['Close'].ewm(span=slow_period, min_periods=1).mean()
    signal_line = macd_line.ewm(span=signal_period, min_periods=1).mean()
    return macd_line, signal_line

def classify_stock(volatility, sharpe_ratio):
    if sharpe_ratio > 1 and volatility < 0.02:
        return "🟢 Invest"
    elif sharpe_ratio > 0 and volatility < 0.05:
        return "🟡 Consider"
    else:
        return "🔴 Avoid"

def get_stock_data_from_db(symbol, start_date=None):
    query = f"SELECT * FROM stocks WHERE symbol='{symbol}'"
    if start_date:
        query += f" AND date >= '{start_date}'"
    return pd.read_sql(query, con=engine)

def save_stock_data_to_db(data):
    data.to_sql("stocks", con=engine, if_exists="append", index=False)

# Streamlit App
st.title("Stock Analysis Dashboard")
st.sidebar.header("Input Options")

# Input: Stock Symbols
symbols = st.sidebar.text_input("Enter Stock Symbols (comma-separated)", "AAPL, MSFT, GOOGL").split(",")
symbols = [symbol.strip().upper() for symbol in symbols]

# Input: Risk-Free Rate
risk_free_rate = st.sidebar.number_input(
    "Risk-Free Rate (Annualized, %)", min_value=0.0, max_value=10.0, value=2.0,
    help="The return you can expect from a risk-free investment (e.g., government bonds)."
) / 252

# Option: Use Max Time Frame
use_max_time_frame = st.sidebar.checkbox("Use Max Available Data", value=False)

# Time Frame Slider
time_frame = st.sidebar.slider(
    "Select Time Frame (Years)", min_value=1, max_value=20, value=2, step=1,
    help="Adjust the time frame for historical stock data (e.g., 1 year, 2 years, Max).",
    disabled=use_max_time_frame
)

# Fetch Data
if st.sidebar.button("Fetch Data"):
    stock_dict = {}
    portfolio_summary = []
    with st.spinner("Fetching stock data..."):
        for symbol in symbols:
            try:
                # Determine start date for filtering
                start_date = None if use_max_time_frame else pd.Timestamp.now() - pd.DateOffset(years=time_frame)

                # Check database for existing data
                existing_data = get_stock_data_from_db(symbol, start_date)
                if not existing_data.empty:
                    st.success(f"Loaded {symbol} data from the database.")
                    data_filtered = existing_data
                else:
                    # Fetch data from Alpha Vantage API
                    data, meta_data = ts.get_daily(symbol=symbol, outputsize="full")
                    data.columns = ["Open", "High", "Low", "Close", "Volume"]
                    data.reset_index(inplace=True)
                    data["symbol"] = symbol
                    save_stock_data_to_db(data)
                    st.success(f"Fetched {symbol} data from API and saved to the database.")
                    data_filtered = data

                # Calculate Metrics
                data_filtered['20_MA'] = data_filtered['Close'].rolling(window=20).mean()
                data_filtered['50_MA'] = data_filtered['Close'].rolling(window=50).mean()
                data_filtered['Upper_Band'] = data_filtered['20_MA'] + (data_filtered['Close'].rolling(window=20).std() * 2)
                data_filtered['Lower_Band'] = data_filtered['20_MA'] - (data_filtered['Close'].rolling(window=20).std() * 2)
                data_filtered['RSI'] = calculate_rsi(data_filtered)
                data_filtered['MACD_Line'], data_filtered['Signal_Line'] = calculate_macd(data_filtered)
                data_filtered['Daily_Return'] = data_filtered['Close'].pct_change()

                # Calculate Sharpe Ratio and Classification
                mean_return = data_filtered['Daily_Return'].mean()
                std_dev = data_filtered['Daily_Return'].std()
                sharpe_ratio = (mean_return - risk_free_rate) / std_dev if std_dev != 0 else 0
                classification = classify_stock(std_dev, sharpe_ratio)

                # Save Summary
                portfolio_summary.append({
                    "Symbol": symbol,
                    "Mean Return (%)": mean_return * 100,
                    "Volatility (%)": std_dev * 100,
                    "Sharpe Ratio": sharpe_ratio,
                    "Classification": classification
                })

                stock_dict[symbol] = data_filtered
                time.sleep(12)  # Respect API rate limits

            except Exception as e:
                st.error(f"Error fetching data for {symbol}: {e}")

    # Portfolio Summary
    if portfolio_summary:
        st.subheader("Portfolio Summary")
        portfolio_df = pd.DataFrame(portfolio_summary)
        st.dataframe(portfolio_df)

    # Plot Data
    if stock_dict:
        for symbol, data in stock_dict.items():
            st.subheader(f"{symbol} Stock Price Analysis")
            fig, ax = plt.subplots(figsize=(14, 7))
            ax.plot(data['Close'], label='Closing Price', alpha=0.75)
            ax.plot(data['20_MA'], label='20-Day Moving Average', alpha=0.75)
            ax.plot(data['50_MA'], label='50-Day Moving Average', alpha=0.75)
            ax.plot(data['Upper_Band'], label='Upper Bollinger Band', linestyle='--')
            ax.plot(data['Lower_Band'], label='Lower Bollinger Band', linestyle='--')
            ax.set_title(f"{symbol} Stock Price with Bollinger Bands")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price (USD)")
            ax.legend()
            st.pyplot(fig)
