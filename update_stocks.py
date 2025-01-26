from alpha_vantage.timeseries import TimeSeries
import mysql.connector
import pandas as pd

# Alpha Vantage API Key
api_key = "GH9ZJMSFZ3HP1RXA"
ts = TimeSeries(key=api_key, output_format="pandas")

# MySQL Connection
def get_db_connection():
    return mysql.connector.connect(
        host="db", user="root", password="password", database="stocks"
    )

def fetch_and_update_stock_data(symbols):
    connection = get_db_connection()
    cursor = connection.cursor()
    for symbol in symbols:
        try:
            data, _ = ts.get_daily(symbol=symbol, outputsize="compact")
            data.reset_index(inplace=True)
            for _, row in data.iterrows():
                cursor.execute("""
                    INSERT INTO stock_data (symbol, date, open_price, high_price, low_price, close_price, volume)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        open_price=VALUES(open_price),
                        high_price=VALUES(high_price),
                        low_price=VALUES(low_price),
                        close_price=VALUES(close_price),
                        volume=VALUES(volume)
                """, (symbol, row['date'], row['1. open'], row['2. high'], row['3. low'], row['4. close'], row['5. volume']))
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
    connection.commit()
    cursor.close()
    connection.close()

def get_all_symbols():
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT portfolio FROM user_portfolios")
    portfolios = cursor.fetchall()
    cursor.close()
    connection.close()
    symbols = set()
    for portfolio in portfolios:
        symbols.update(portfolio[0].split(","))
    return list(symbols)

if __name__ == "__main__":
    symbols = get_all_symbols()
    fetch_and_update_stock_data(symbols)
