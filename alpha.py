from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import matplotlib.pyplot as plt
import time
import streamlit as st

# Alpha Vantage API key
api_key = "GH9ZJMSFZ3HP1RXA"

# Initialize Alpha Vantage TimeSeries object
ts = TimeSeries(key=api_key, output_format="pandas")

# Helper Functions
def calculate_rsi(data, window=14):
    """Calculate Relative Strength Index (RSI)."""
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    """Calculate MACD and Signal Line."""
    macd_line = data['Close'].ewm(span=fast_period, min_periods=1).mean() - \
                data['Close'].ewm(span=slow_period, min_periods=1).mean()
    signal_line = macd_line.ewm(span=signal_period, min_periods=1).mean()
    return macd_line, signal_line

def classify_stock(volatility, sharpe_ratio):
    """Classify stocks based on volatility and Sharpe Ratio."""
    if sharpe_ratio > 1 and volatility < 0.02:
        return "🟢 Invest"
    elif sharpe_ratio > 0 and volatility < 0.05:
        return "🟡 Consider"
    else:
        return "🔴 Avoid"

# Streamlit app layout
st.title("Stock Analysis Dashboard")
st.sidebar.header("Input Options")

# Sidebar: Input for stock symbols
symbols = st.sidebar.text_input("Enter Stock Symbols (comma-separated)", "AAPL, MSFT, GOOGL").split(",")
symbols = [symbol.strip().upper() for symbol in symbols]  # Clean input

# Sidebar: Risk-Free Rate (with tooltip explanation)
risk_free_rate = st.sidebar.number_input(
    "Risk-Free Rate (Annualized, %)",
    min_value=0.0,
    max_value=10.0,
    value=2.0,
    help="The return you can expect from a risk-free investment (e.g., government bonds)."
) / 252  # Convert to daily

# Sidebar: Time Frame Slider
time_frame = st.sidebar.slider(
    "Select Time Frame (Years)",
    min_value=1,
    max_value=20,
    value=2,
    step=1,
    help="Adjust the time frame for historical stock data (e.g., 1 year, 2 years, Max)."
)
# Option to use the max time frame
use_max_time_frame = st.sidebar.checkbox("Use Max Available Data", value=False)



# Sidebar: Fetch data button
if st.sidebar.button("Fetch Data"):
    stock_dict = {}
    portfolio_summary = []
    with st.spinner("Fetching stock data..."):
        for symbol in symbols:
            try:
                # Fetch stock data
                data, meta_data = ts.get_daily(symbol=symbol, outputsize="full")
                data.columns = ["Open", "High", "Low", "Close", "Volume"]
                
                # Filter data based on time frame
                if use_max_time_frame:
                    data_filtered = data  # Use all available data
                else:
                    data_filtered = data.iloc[-time_frame * 252:]  # Approx. 252 trading days per year

                # Calculate metrics
                data['20_MA'] = data['Close'].rolling(window=20).mean()
                data['50_MA'] = data['Close'].rolling(window=50).mean()
                data['Upper_Band'] = data['20_MA'] + (data['Close'].rolling(window=20).std() * 2)
                data['Lower_Band'] = data['20_MA'] - (data['Close'].rolling(window=20).std() * 2)
                data['RSI'] = calculate_rsi(data)
                data['MACD_Line'], data['Signal_Line'] = calculate_macd(data)
                data[['RSI', 'MACD_Line', 'Signal_Line']] = data[['RSI', 'MACD_Line', 'Signal_Line']].fillna(0)
                data['Daily_Return'] = data['Close'].pct_change()

                # Calculate Sharpe Ratio
                mean_return = data['Daily_Return'].mean()
                std_dev = data['Daily_Return'].std()
                sharpe_ratio = (mean_return - risk_free_rate) / std_dev if std_dev != 0 else 0

                # Classify stock based on metrics
                classification = classify_stock(std_dev, sharpe_ratio)
                
                # Save data to dictionary
                stock_dict[symbol] = data

                # Add to portfolio summary
                portfolio_summary.append({
                    "Symbol": symbol,
                    "Mean Return (%)": mean_return * 100,
                    "Volatility (%)": std_dev * 100,
                    "Sharpe Ratio": sharpe_ratio,
                    "Classification": classification
                })
                
                st.success(f"Fetched data for {symbol}.")
                time.sleep(12)  # To respect API rate limits
            except Exception as e:
                st.error(f"Error fetching data for {symbol}: {e}")
    
    # Display portfolio summary
    if portfolio_summary:
        st.subheader("Portfolio Summary")
        portfolio_df = pd.DataFrame(portfolio_summary)
        st.dataframe(portfolio_df)

    # Plot the data for each stock
    if stock_dict:
        for symbol, data in stock_dict.items():
            st.subheader(f"{symbol} Stock Price Analysis")

            # Plot the stock data with Bollinger Bands
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

            # Plot RSI
            st.subheader(f"{symbol} RSI")
            fig, ax = plt.subplots(figsize=(14, 4))
            ax.plot(data['RSI'], label='RSI', color='blue')
            ax.axhline(70, color='red', linestyle='--', label='Overbought (70)')
            ax.axhline(30, color='green', linestyle='--', label='Oversold (30)')
            ax.set_title(f"{symbol} RSI")
            ax.legend()
            st.pyplot(fig)

            # Plot MACD
            st.subheader(f"{symbol} MACD")
            fig, ax = plt.subplots(figsize=(14, 4))
            ax.plot(data['MACD_Line'], label='MACD Line', color='blue')
            ax.plot(data['Signal_Line'], label='Signal Line', color='red')
            ax.axhline(0, color='black', linestyle='--', label='Zero Line')
            ax.set_title(f"{symbol} MACD")
            ax.legend()
            st.pyplot(fig)
