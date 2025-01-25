from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import matplotlib.pyplot as plt
import time
import streamlit as st

# Replace with your Alpha Vantage API key
api_key = "GH9ZJMSFZ3HP1RXA"

# Initialize Alpha Vantage TimeSeries object
ts = TimeSeries(key=api_key, output_format="pandas")

# Streamlit app layout
st.title("Stock Analysis Dashboard")
st.sidebar.header("Input Options")

# Sidebar: Input for stock symbol
symbols = st.sidebar.text_input("Enter Stock Symbols (comma-separated)", "AAPL, MSFT, GOOGL").split(",")
symbols = [symbol.strip().upper() for symbol in symbols]  # Clean input

# Sidebar: Date range (currently not supported by Alpha Vantage)
st.sidebar.write("Data range: Full historical data (Alpha Vantage free tier)")

# Sidebar: Fetch data button
if st.sidebar.button("Fetch Data"):
    stock_dict = {}
    with st.spinner("Fetching stock data..."):
        for symbol in symbols:
            try:
                # Fetch stock data
                data, meta_data = ts.get_daily(symbol=symbol, outputsize="full")
                data.columns = ["Open", "High", "Low", "Close", "Volume"]
                
                # Calculate metrics
                data['20_MA'] = data['Close'].rolling(window=20).mean()
                data['50_MA'] = data['Close'].rolling(window=50).mean()
                data['Upper_Band'] = data['20_MA'] + (data['Close'].rolling(window=20).std() * 2)
                data['Lower_Band'] = data['20_MA'] - (data['Close'].rolling(window=20).std() * 2)
                
                # Save to dictionary
                stock_dict[symbol] = data
                
                st.success(f"Fetched data for {symbol}.")
                time.sleep(12)  # To respect API rate limits
            except Exception as e:
                st.error(f"Error fetching data for {symbol}: {e}")
    
    # Plot the data for each stock
    if stock_dict:
        for symbol, data in stock_dict.items():
            st.subheader(f"{symbol} Stock Price Analysis")
            
            # Plot the data
            fig, ax = plt.subplots(figsize=(14, 7))
            ax.plot(data['Close'], label='Closing Price', alpha=0.75)
            ax.plot(data['20_MA'], label='20-Day Moving Average', alpha=0.75)
            ax.plot(data['50_MA'], label='50-Day Moving Average', alpha=0.75)
            ax.plot(data['Upper_Band'], label='Upper Bollinger Band', linestyle='--')
            ax.plot(data['Lower_Band'], label='Lower Bollinger Band', linestyle='--')

            # Customize plot
            ax.set_title(f"{symbol} Stock Price with Bollinger Bands")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price (USD)")
            ax.legend()
            
            # Display the plot in Streamlit
            st.pyplot(fig)

            # Download processed data as CSV
            csv = data.to_csv().encode('utf-8')
            st.download_button(
                label=f"Download {symbol} Data as CSV",
                data=csv,
                file_name=f"{symbol}_stock_data.csv",
                mime='text/csv'
            )

