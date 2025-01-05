import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import spearmanr

# Function to read the sentiment and stock data
def read_data(sentiment_file, stock_file):
    # Read sentiment data
    sentiment_data = pd.read_csv(sentiment_file, parse_dates=['Date'])
    
    # Read stock data
    stock_data = pd.read_csv(stock_file, parse_dates=['Date'])
    
    # Convert stock 'Date' column to date-only format to match sentiment data
    stock_data['Date'] = stock_data['Date'].dt.date
    
    # Ensure sentiment data 'Date' is in date format (without time)
    sentiment_data['Date'] = pd.to_datetime(sentiment_data['Date']).dt.date
    
    # Fill missing values with forward fill, then backward fill
    sentiment_data['New_Sentiment'] = sentiment_data['New_Sentiment'].fillna(method='ffill').fillna(method='bfill')
    stock_data['Close'] = stock_data['Close'].fillna(method='ffill').fillna(method='bfill')
    
    return sentiment_data, stock_data

# Function to calculate Spearman's rank correlation
def calculate_spearman(sentiment_file, stock_file):
    # Read the data
    sentiment_data, stock_data = read_data(sentiment_file, stock_file)
    
    # Print unique dates in both datasets for debugging
    print("Sentiment data unique dates:", sentiment_data['Date'].unique())
    print("Stock data unique dates:", stock_data['Date'].unique())
    
    # Merge sentiment and stock data on the 'Date' column
    merged_data = pd.merge(sentiment_data, stock_data, how='inner', on='Date')
    
    # If no matching dates found, notify the user and return
    if merged_data.empty:
        print("No matching dates found after merging.")
        return
    
    # Extract sentiment and stock values
    sentiment_trend = merged_data['New_Sentiment'].values
    stock_trend = merged_data['Close'].values
    
    # Print the shapes of the trends
    print(f"Sentiment trend shape: {sentiment_trend.shape}")
    print(f"Stock trend shape: {stock_trend.shape}")
    
    # Calculate Spearman's rank correlation
    correlation, p_value = spearmanr(sentiment_trend, stock_trend)
    
    # Print the Spearman correlation and p-value
    print(f"Spearman Rank Correlation: {correlation}")
    print(f"P-value: {p_value}")

if __name__ == '__main__':
    # Run the function with your file paths
    calculate_spearman("./data/91293_sentiment_model.csv", "./data/HXE.csv")
