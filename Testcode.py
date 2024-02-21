import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from textblob import TextBlob
import ccxt

# Function to fetch historical cryptocurrency data
def fetch_historical_data(symbol, timeframe='1d', limit=100):
    exchange = ccxt.binance()  # Example: Using Binance exchange
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df.set_index('timestamp')

# Function to calculate sentiment score from news headlines
def calculate_sentiment_score(headlines):
    sentiment_score = 0
    for headline in headlines:
        blob = TextBlob(headline)
        sentiment_score += blob.sentiment.polarity
    return sentiment_score / len(headlines)

# Function to generate trading signals based on AI analysis
def generate_trading_signals(symbol, timeframe='1d', limit=100):
    # Fetch historical data
    data = fetch_historical_data(symbol, timeframe, limit)
    
    # Calculate moving averages
    data['MA_50'] = data['close'].rolling(window=50).mean()
    data['MA_200'] = data['close'].rolling(window=200).mean()
    
    # Perform sentiment analysis on news headlines
    headlines = ['Bitcoin price hits all-time high', 'Ethereum upgrades upcoming', 'Crypto regulation news']
    sentiment_score = calculate_sentiment_score(headlines)
    
    # Feature Engineering
    data['Price Change'] = data['close'].pct_change()
    data['Volume Change'] = data['volume'].pct_change()
    
    # Drop NaN values
    data.dropna(inplace=True)
    
    # Define features and target variable
    X = data[['MA_50', 'MA_200', 'Price Change', 'Volume Change']]
    y = np.where(data['close'].shift(-1) > data['close'], 1, -1)  # 1 for buy, -1 for sell
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Predictions
    predictions = clf.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    
    # Generate trading signal based on AI analysis
    last_data = X.iloc[-1].values.reshape(1, -1)
    signal = clf.predict(last_data)[0]
    if signal == 1:
        return 'Buy'
    elif signal == -1:
        return 'Sell'
    else:
        return 'Hold'

# Example usage
symbol = 'BTC/USDT'
timeframe = '1d'
limit = 100
signal = generate_trading_signals(symbol, timeframe, limit)
print(f'Trading signal for {symbol} based on AI analysis: {signal}')
