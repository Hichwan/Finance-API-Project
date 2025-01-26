CREATE TABLE IF NOT EXISTS user_portfolios (
    user_id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(255) NOT NULL,
    portfolio TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS stock_data (
    symbol VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    open_price FLOAT,
    high_price FLOAT,
    low_price FLOAT,
    close_price FLOAT,
    volume INT,
    PRIMARY KEY (symbol, date)
);
