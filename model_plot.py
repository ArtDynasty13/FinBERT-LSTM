import pandas as pd
import matplotlib.pyplot as plt

def plot_sentiment_stock(start_date=None, end_date=None):
    # Load the sentiment and stock price data
    sentiment_file_path = "./91293_sentiment_model.csv"
    stock_file_path = "./HXE.csv"

    sentiment_data = pd.read_csv(sentiment_file_path)
    stock_data = pd.read_csv(stock_file_path)

    # Convert 'Date' columns to datetime and remove time (if necessary)
    sentiment_data['Date'] = pd.to_datetime(sentiment_data['Date']).dt.date  # Keeping only the date part
    stock_data['Date'] = pd.to_datetime(stock_data['Date']).dt.date  # Keeping only the date part

    # Filter the sentiment data by custom date range if specified
    if start_date and end_date:
        sentiment_data_filtered = sentiment_data[(sentiment_data['Date'] >= start_date) & (sentiment_data['Date'] <= end_date)]
    else:
        sentiment_data_filtered = sentiment_data

    # Merge sentiment and stock data on the 'Date' column
    merged_data = pd.merge(sentiment_data_filtered, stock_data, on="Date", how="left")

    # Carry forward the stock prices (forward fill)
    merged_data['Close'] = merged_data['Close'].fillna(method='ffill')

    # Plot the sentiment and stock data on the same graph
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot the sentiment dynamics on the primary y-axis (left)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Sentiment', color='b')
    ax1.plot(merged_data['Date'], merged_data['New_Sentiment'], color='b', label='Sentiment Dynamics')
    ax1.tick_params(axis='y', labelcolor='b')

    # Create a secondary y-axis to plot the stock prices
    ax2 = ax1.twinx()
    ax2.set_ylabel('Stock Close Price ($)', color='g')
    ax2.plot(merged_data['Date'], merged_data['Close'], color='g', label='Stock Close Price')
    ax2.tick_params(axis='y', labelcolor='g')

    # Dynamically adjust the y-axis limits based on data
    ax1.set_ylim(merged_data['New_Sentiment'].min() - 1, merged_data['New_Sentiment'].max() + 1)
    ax2.set_ylim(merged_data['Close'].min() - 5, merged_data['Close'].max() + 5)

    # Add titles and labels
    plt.title('Sentiment Dynamics and Stock Close Price Over Time')

    # Format the x-axis to show readable date labels
    plt.xticks(rotation=45)

    # Display the plot
    fig.tight_layout()
    plt.show()
