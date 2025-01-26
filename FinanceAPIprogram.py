from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import matplotlib.pyplot as plt
import time
import streamlit as st
from sqlalchemy import create_engine, text
import os

# MySQL Connection Details (Environment Variables for Docker/Kubernetes)
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "password")
MYSQL_HOST = os.getenv("MYSQL_HOST", "db")
MYSQL_PORT = os.getenv("MYSQL_PORT", 3306)
MYSQL_DB = os.getenv("MYSQL_DB", "stocks")
DATABASE_URL = f"mysql+mysqlconnector://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"

# Set up SQLAlchemy Engine
engine = create_engine(DATABASE_URL)

# Alpha Vantage API Key
api_key = "GH9ZJMSFZ3HP1RXA"
ts = TimeSeries(key=api_key, output_format="pandas")


# Helper Functions
def validate_env_vars():
    if not all([MYSQL_USER, MYSQL_PASSWORD, MYSQL_HOST, MYSQL_DB]):
        raise ValueError("Missing required MySQL environment variables!")


validate_env_vars()


def connect_to_db():
    try:
        connection = engine.connect()
        print("Database connection successful!")
        return connection
    except Exception as e:
        print(f"Database connection failed: {e}")
        raise


def calculate_rsi(data, window=14):
    delta = data["Close"].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    macd_line = (
        data["Close"].ewm(span=fast_period, min_periods=1).mean()
        - data["Close"].ewm(span=slow_period, min_periods=1).mean()
    )
    signal_line = macd_line.ewm(span=signal_period, min_periods=1).mean()
    return macd_line, signal_line

def calculate_true_sharpe_ratio(data, risk_free_rate, time_frame_years):
    """
    Calculate the true Sharpe Ratio for the selected time frame.
    Args:
        data (DataFrame): Data with 'Daily_Return'.
        risk_free_rate (float): Annualized risk-free rate.
        time_frame_years (float): Time frame in years.
    Returns:
        float: Sharpe Ratio for the selected period.
    """
    # Filter data for the selected time frame
    period_returns = data["Daily_Return"].dropna()
    
    # Calculate the mean and standard deviation of returns for the period
    avg_return_period = period_returns.mean() * (252 * time_frame_years)  # Scale to time frame
    std_dev_period = period_returns.std() * (252**0.5)  # Annualized volatility

    # Adjust the risk-free rate to the selected time frame
    adjusted_risk_free_rate = risk_free_rate * time_frame_years

    # Calculate Sharpe Ratio
    sharpe_ratio = (avg_return_period - adjusted_risk_free_rate) / std_dev_period if std_dev_period != 0 else 0
    return sharpe_ratio


def classify_stock(volatility, sharpe_ratio_daily, sharpe_ratio_cumulative, risk_tolerance):
    """
    Classify stock based on both short-term (daily Sharpe ratio) and long-term (cumulative Sharpe ratio).
    """
    classification = {}

    # Short-term classification (daily Sharpe ratio)
    if risk_tolerance == "High":
        if sharpe_ratio_daily > 0.5 and volatility < 0.1:
            classification["Short-Term"] = "\U0001F7E2 Invest"  # Green
        elif sharpe_ratio_daily > 0 and volatility < 0.2:
            classification["Short-Term"] = "\U0001F7E1 Consider"  # Yellow
        else:
            classification["Short-Term"] = "\U0001F534 Avoid"  # Red
    elif risk_tolerance == "Moderate":
        if sharpe_ratio_daily > 1 and volatility < 0.05:
            classification["Short-Term"] = "\U0001F7E2 Invest"
        elif sharpe_ratio_daily > 0 and volatility < 0.1:
            classification["Short-Term"] = "\U0001F7E1 Consider"
        else:
            classification["Short-Term"] = "\U0001F534 Avoid"
    elif risk_tolerance == "Low":
        if sharpe_ratio_daily > 1.5 and volatility < 0.03:
            classification["Short-Term"] = "\U0001F7E2 Invest"
        elif sharpe_ratio_daily > 0.5 and volatility < 0.05:
            classification["Short-Term"] = "\U0001F7E1 Consider"
        else:
            classification["Short-Term"] = "\U0001F534 Avoid"

    # Long-term classification (cumulative Sharpe ratio)
    if risk_tolerance == "High":
        if sharpe_ratio_cumulative > 1 and volatility < 0.2:
            classification["Long-Term"] = "\U0001F7E2 Invest"  # Green
        elif sharpe_ratio_cumulative > 0.5 and volatility < 0.3:
            classification["Long-Term"] = "\U0001F7E1 Consider"  # Yellow
        else:
            classification["Long-Term"] = "\U0001F534 Avoid"  # Red
    elif risk_tolerance == "Moderate":
        if sharpe_ratio_cumulative > 1.5 and volatility < 0.1:
            classification["Long-Term"] = "\U0001F7E2 Invest"
        elif sharpe_ratio_cumulative > 0.5 and volatility < 0.2:
            classification["Long-Term"] = "\U0001F7E1 Consider"
        else:
            classification["Long-Term"] = "\U0001F534 Avoid"
    elif risk_tolerance == "Low":
        if sharpe_ratio_cumulative > 2 and volatility < 0.05:
            classification["Long-Term"] = "\U0001F7E2 Invest"
        elif sharpe_ratio_cumulative > 1 and volatility < 0.1:
            classification["Long-Term"] = "\U0001F7E1 Consider"
        else:
            classification["Long-Term"] = "\U0001F534 Avoid"

    return classification


def get_stock_data_from_db(symbol, start_date=None):
    query = f"SELECT * FROM stock_data WHERE symbol='{symbol}'"
    if start_date:
        query += f" AND date >= '{start_date}'"
    query += " ORDER BY date DESC"
    df = pd.read_sql(query, con=engine, parse_dates=["date"])
    # Rename columns back to match the expected structure
    df.rename(
        columns={
            "open_price": "Open",
            "high_price": "High",
            "low_price": "Low",
            "close_price": "Close",
            "volume": "Volume",
        },
        inplace=True,
    )
    return df


def save_stock_data_to_db(data):
    # Rename columns to match the database schema
    data["date"] = pd.to_datetime(data["date"])
    data.rename(
        columns={
            "Open": "open_price",
            "High": "high_price",
            "Low": "low_price",
            "Close": "close_price",
            "Volume": "volume",
        },
        inplace=True
    )
    data.to_sql("stock_data", con=engine, if_exists="append", index=False)


def save_portfolio(username, portfolio):
    connection = connect_to_db()
    try:
        query = text(
            """
            INSERT INTO user_portfolios (username, portfolio)
            VALUES (:username, :portfolio)
            ON DUPLICATE KEY UPDATE portfolio=VALUES(portfolio);
        """
        )
        connection.execute(query, {"username": username, "portfolio": ",".join(portfolio)})
        print(f"Saved portfolio for {username}: {portfolio}")
    finally:
        connection.close()


def get_portfolio(username):
    connection = connect_to_db()
    try:
        query = text("SELECT portfolio FROM user_portfolios WHERE TRIM(username)=:username")
        result = connection.execute(query, {"username": username}).fetchone()
    finally:
        connection.close()
    return result[0].split(",") if result else None


def clear_portfolio(username):
    connection = connect_to_db()
    try:
        query = text("DELETE FROM user_portfolios WHERE username=:username")
        connection.execute(query, {"username": username})
        print(f"Cleared portfolio for {username}")
    finally:
        connection.close()


@st.cache_data(ttl=3600)
def load_symbol_list():
    symbols_df = pd.read_csv("listing_status.csv")
    return symbols_df["Symbol"].tolist()


def validate_symbols(symbols):
    valid_symbols = load_symbol_list()
    return [symbol for symbol in symbols if symbol in valid_symbols]


# Streamlit App
st.title("Stock Analysis Dashboard")
st.sidebar.header("Input Options")

username = st.sidebar.text_input("Enter your username", help="Enter your unique username to save your portfolio.")

valid_symbol_list = load_symbol_list()

symbols = st.sidebar.multiselect(
    "Select Stocks for Your Portfolio",
    valid_symbol_list,
    help="Start typing to find and select stock symbols (e.g., AAPL, MSFT).",
)

if st.sidebar.button("Save Portfolio"):
    if symbols:
        save_portfolio(username, symbols)
        st.success("Portfolio saved successfully!")
    else:
        st.error("Please select at least one stock symbol to save.")

if st.sidebar.button("Load Portfolio"):
    portfolio = get_portfolio(username)
    if portfolio:
        st.write(f"Your Portfolio: {', '.join(portfolio)}")
    else:
        st.error("No portfolio found for this username.")

if st.sidebar.button("Clear Portfolio"):
    clear_portfolio(username)
    st.success("Portfolio cleared!")

risk_free_rate = st.sidebar.number_input(
    "Risk-Free Rate (Annualized, %)", min_value=0.0, max_value=10.0, value=2.0,
    help="The return you can expect from a risk-free investment (e.g., government bonds)."
) / 252

# Add risk tolerance option
risk_tolerance = st.sidebar.selectbox(
    "Select Your Risk Tolerance",
    options=["High", "Moderate", "Low"],
    help="High: Willing to take on more risk for higher returns. Moderate: Balanced risk/return. Low: Prefer safer investments."
)

use_max_time_frame = st.sidebar.checkbox("Use Max Available Data", value=False)

time_frame = st.sidebar.slider(
    "Select Time Frame (Years)", min_value=1, max_value=20, value=2, step=1,
    help="Adjust the time frame for historical stock data (e.g., 1 year, 2 years, Max).",
    disabled=use_max_time_frame,
)

stock_dict = {}

if st.sidebar.button("Fetch Data"): 
    portfolio_summary_metrics = []  # For numerical data
    portfolio_summary_recommendations = []  # For classifications
    with st.spinner("Fetching stock data..."):
        for symbol in symbols:
            try:
                # Load or fetch data
                start_date = None if use_max_time_frame else pd.Timestamp.now() - pd.DateOffset(years=time_frame)
                existing_data = get_stock_data_from_db(symbol, start_date)
                if not existing_data.empty:
                    st.success(f"Loaded {symbol} data from the database.")
                    data_filtered = existing_data
                else:
                    data, meta_data = ts.get_daily(symbol=symbol, outputsize="full")
                    data.columns = ["Open", "High", "Low", "Close", "Volume"]
                    data.reset_index(inplace=True)
                    data["date"] = pd.to_datetime(data["date"])
                    data["symbol"] = symbol
                    save_stock_data_to_db(data)
                    st.success(f"Fetched {symbol} data from API and saved to the database.")
                    data_filtered = data

                # Set date as index
                data_filtered["date"] = pd.to_datetime(data_filtered["date"])
                data_filtered.set_index("date", inplace=True)

                # Calculate metrics
                if len(data_filtered) > 14:  # Ensure sufficient data for RSI and MACD
                    data_filtered["20_MA"] = data_filtered["Close"].rolling(window=20).mean()
                    data_filtered["50_MA"] = data_filtered["Close"].rolling(window=50).mean()
                    data_filtered["Upper_Band"] = data_filtered["20_MA"] + (data_filtered["Close"].rolling(window=20).std() * 2)
                    data_filtered["Lower_Band"] = data_filtered["20_MA"] - (data_filtered["Close"].rolling(window=20).std() * 2)
                    data_filtered["RSI"] = calculate_rsi(data_filtered)
                    data_filtered["MACD_Line"], data_filtered["Signal_Line"] = calculate_macd(data_filtered)
                    data_filtered["Daily_Return"] = data_filtered["Close"].pct_change()

                    # Sharpe Ratios
                    sharpe_ratio_daily = (data_filtered["Daily_Return"].mean() - risk_free_rate) / data_filtered["Daily_Return"].std() if data_filtered["Daily_Return"].std() != 0 else 0
                    sharpe_ratio_cumulative = calculate_true_sharpe_ratio(data_filtered, risk_free_rate, time_frame)

                    # Classification
                    classification = classify_stock(
                        data_filtered["Daily_Return"].std(),
                        sharpe_ratio_daily,
                        sharpe_ratio_cumulative,
                        risk_tolerance
                    )

                    # Append metrics and recommendations to their respective summaries
                    portfolio_summary_metrics.append({
                        "Symbol": symbol,
                        "Mean Return (Daily, %)": data_filtered["Daily_Return"].mean() * 100,
                        "Volatility (Daily, %)": data_filtered["Daily_Return"].std() * 100,
                        "Sharpe Ratio (Daily)": sharpe_ratio_daily,
                        "Sharpe Ratio (Cumulative)": sharpe_ratio_cumulative,
                    })
                    portfolio_summary_recommendations.append({
                        "Symbol": symbol,
                        "Short-Term Recommendation": classification.get("Short-Term"),
                        "Long-Term Recommendation": classification.get("Long-Term"),
                    })

                    stock_dict[symbol] = data_filtered
                else:
                    st.warning(f"Insufficient data for {symbol} to calculate RSI or MACD.")

                time.sleep(12)

            except Exception as e:
                st.error(f"Error fetching data for {symbol}: {e}")

    # Display results
    if portfolio_summary_metrics:
        st.subheader("Portfolio Metrics")
        portfolio_metrics_df = pd.DataFrame(portfolio_summary_metrics)
        st.dataframe(portfolio_metrics_df)

    if portfolio_summary_recommendations:
        st.subheader("Portfolio Recommendations")
        portfolio_recommendations_df = pd.DataFrame(portfolio_summary_recommendations)
        st.dataframe(portfolio_recommendations_df)


if stock_dict:
    for symbol, data in stock_dict.items():
        st.subheader(f"{symbol} Stock Price Analysis")

        # Debugging: Display the dataset
        st.write(f"Data for {symbol}:")
        st.write(data.head())

        # Ensure the date column is set as the index
        if "date" in data.columns:
            data["date"] = pd.to_datetime(data["date"], errors="coerce")  # Convert to datetime
            data = data.dropna(subset=["date"])  # Drop rows with invalid dates
            data.set_index("date", inplace=True)  # Set date as index

        # Check required columns
        required_columns = ["Close", "20_MA", "50_MA", "Upper_Band", "Lower_Band", "RSI", "MACD_Line", "Signal_Line"]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            st.error(f"Missing required columns for {symbol}: {missing_columns}")
            continue

        # Bollinger Bands and Closing Price
        with st.expander(f"About Bollinger Bands and Closing Price for {symbol}"):
            st.markdown(
                """
                - **Bollinger Bands**: Used to measure market volatility and identify overbought or oversold conditions.
                - The **Upper Band** represents a resistance level, while the **Lower Band** represents a support level.
                - **Closing Price**: The price at which the stock closed for the day.
                """
            )

        # Calculate both short-term and long-term Sharpe Ratios
        sharpe_ratio_daily = (data["Daily_Return"].mean() - risk_free_rate) / data["Daily_Return"].std() if data["Daily_Return"].std() != 0 else 0
        sharpe_ratio_cumulative = calculate_true_sharpe_ratio(data, risk_free_rate, time_frame)

        # Plot Bollinger Bands and Closing Price
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(data.index, data["Close"], label="Closing Price", alpha=0.75)
        ax.plot(data.index, data["20_MA"], label="20-Day Moving Average", alpha=0.75)
        ax.plot(data.index, data["50_MA"], label="50-Day Moving Average", alpha=0.75)
        ax.fill_between(data.index, data["Lower_Band"], data["Upper_Band"], alpha=0.2, label="Bollinger Bands")

        # Add Sharpe Ratios to the title
        ax.set_title(f"{symbol} Stock Price with Bollinger Bands\n"
                    f"Sharpe Ratio (Daily): {sharpe_ratio_daily:.2f} | Sharpe Ratio (Cumulative): {sharpe_ratio_cumulative:.2f}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.legend()
        st.pyplot(fig)


        # RSI
        with st.expander(f"About RSI for {symbol}"):
            st.markdown(
                """
                - **Relative Strength Index (RSI)**: Measures the strength and momentum of price movements.
                - Values above 70 indicate overbought conditions, suggesting a potential price decline.
                - Values below 30 indicate oversold conditions, suggesting a potential price increase.
                """
            )

        # Plot RSI
        st.subheader(f"{symbol} RSI")
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(data.index, data['RSI'], label='RSI', color='blue')
        ax.axhline(70, color='red', linestyle='--', label='Overbought (70)')
        ax.axhline(30, color='green', linestyle='--', label='Oversold (30)')
        ax.set_title(f"{symbol} RSI")
        ax.set_xlabel("Date")
        ax.legend()
        st.pyplot(fig)

        # MACD
        with st.expander(f"About MACD for {symbol}"):
            st.markdown(
                """
                - **MACD (Moving Average Convergence Divergence)**: Indicates trends and momentum.
                - The **MACD Line** is calculated as the difference between the fast and slow exponential moving averages (EMAs).
                - The **Signal Line** is the EMA of the MACD Line and helps identify buy/sell signals.
                - Positive values above 0 suggest upward momentum, while negative values below 0 suggest downward momentum.
                """
            )

        # Plot MACD
        st.subheader(f"{symbol} MACD")
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(data.index, data['MACD_Line'], label='MACD Line', color='blue')
        ax.plot(data.index, data['Signal_Line'], label='Signal Line', color='red')
        ax.axhline(0, color='black', linestyle='--', label='Zero Line')
        ax.set_title(f"{symbol} MACD")
        ax.set_xlabel("Date")
        ax.legend()
        st.pyplot(fig)
