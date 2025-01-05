import pandas as pd
from sklearn.metrics import mean_squared_error

# Function to calculate daily sentiment dynamics and return MSE
def calculate_daily_sentiment(data, nudge_strength, neut_weight, stock_file):
    current_sentiment = 0  # Initial sentiment value
    daily_sentiment_data = []

    for date, group in data.groupby('PubDate'):
        # Count positive, neutral, and negative sentiment for the day
        pos_count = group[group['sentiment_value'] == 1].shape[0]
        neut_count = group[group['sentiment_value'] == 0].shape[0]
        neg_count = group[group['sentiment_value'] == -1].shape[0]

        # Calculate the new sentiment based on the previous sentiment
        new_sentiment = current_sentiment
        new_sentiment += pos_count * nudge_strength
        new_sentiment -= neg_count * nudge_strength
        new_sentiment += neut_count * nudge_strength * (-neut_weight if new_sentiment > 0 else neut_weight)

        # Store the results for the day
        daily_sentiment_data.append({
            'Date': date,
            'New_Sentiment': new_sentiment
        })

        # Update the current sentiment
        current_sentiment = new_sentiment

    # Convert daily sentiment data to DataFrame
    daily_sentiment_df = pd.DataFrame(daily_sentiment_data)

    # Merge with stock data
    stock_data = pd.read_csv(stock_file)
    stock_data['Date'] = pd.to_datetime(stock_data['Date']).dt.date
    daily_sentiment_df['Date'] = pd.to_datetime(daily_sentiment_df['Date']).dt.date
    merged_data = pd.merge(daily_sentiment_df, stock_data, on='Date', how='left')

    # Handle missing values
    merged_data['Close'] = merged_data['Close'].fillna(method='ffill').fillna(method='bfill')
    merged_data['New_Sentiment'] = merged_data['New_Sentiment'].fillna(method='ffill').fillna(method='bfill')

    # Calculate and return MSE
    mse = mean_squared_error(merged_data['Close'], merged_data['New_Sentiment'])
    return mse

# Load the data
file_path = "./data/91293_sentiment.csv"
stock_file = "./data/HXE.csv"
data = pd.read_csv(file_path)

# Convert 'PubDate' to datetime
data['PubDate'] = pd.to_datetime(data['PubDate'])
data['PubDate'] = data['PubDate'].dt.date

# Map sentiment labels to numerical values (-1, 0, 1)
sentiment_mapping = {'positive': 1, 'neutral': 0, 'negative': -1}
data['sentiment_value'] = data['sentiment'].map(sentiment_mapping)

# Define ranges for optimization
nudge_strength = 0.1
neut_weights = [i * 0.05 for i in range(1, 21)]  # Test values from 0.05 to 1.0
mse_results = []

# Optimize neut_weight
for neut_weight in neut_weights:
    mse = calculate_daily_sentiment(data, nudge_strength, neut_weight, stock_file)
    mse_results.append({'neut_weight': neut_weight, 'mse': mse})
    print(f"neut_weight: {neut_weight:.2f}, MSE: {mse:.4f}")

# Find the best neut_weight
optimal_result = min(mse_results, key=lambda x: x['mse'])
optimal_neut_weight = optimal_result['neut_weight']
optimal_mse = optimal_result['mse']

print(f"\nOptimal neut_weight: {optimal_neut_weight:.2f} with MSE: {optimal_mse:.4f}")
