# Stock Analysis Dashboard

The **Stock Analysis Dashboard** is a comprehensive web application built using **Streamlit**, **TensorFlow**, and **SQLAlchemy** to help users analyze stock market trends, visualize historical data, and predict future stock prices using **LSTM (Long Short-Term Memory)** neural networks.

## Features

- **Portfolio Management**:
  - Save, load, and clear your custom stock portfolios.
- **Stock Visualizations**:
  - **Bollinger Bands**: Visualize market volatility.
  - **RSI (Relative Strength Index)**: Identify overbought/oversold conditions.
  - **MACD (Moving Average Convergence Divergence)**: Identify trends and momentum.
- **LSTM Predictions**:
  - Predict future stock prices using LSTM models.
  - Evaluate model performance with **Root Mean Square Error (RMSE)**.
- **SQL Database Integration**:
  - Store and retrieve stock data efficiently using MySQL.
- **Alpha Vantage Integration**:
  - Fetch up-to-date stock market data using the Alpha Vantage API.

---

## Technologies Used

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Backend**: Python
- **Machine Learning**: TensorFlow
- **Database**: MySQL (via SQLAlchemy)
- **Data Source**: [Alpha Vantage API](https://www.alphavantage.co/)
- **Visualization**: Matplotlib

---

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- MySQL installed locally or on a server
- Alpha Vantage API key (free to obtain)

### Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/stock-analysis-dashboard.git
   cd stock-analysis-dashboard
