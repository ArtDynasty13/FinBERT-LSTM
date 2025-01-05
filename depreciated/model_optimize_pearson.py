import pandas as pd
from scipy.stats import pearsonr

# Function to calculate daily sentiment dynamics and return Pearson correlation
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

    # Calculate Pearson correlation and return it
    correlation, _ = pearsonr(merged_data['Close'], merged_data['New_Sentiment'])
    return correlation

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

# Define ranges for optimization (testing values from 0.15 to 0.25 in 0.01 increments)
nudge_strength = 0.1
neut_weights = [i * 0.01 + 0.15 for i in range(11)]  # Values from 0.15 to 0.25
correlation_results = []

# Optimize neut_weight based on Pearson correlation
for neut_weight in neut_weights:
    correlation = calculate_daily_sentiment(data, nudge_strength, neut_weight, stock_file)
    correlation_results.append({'neut_weight': neut_weight, 'correlation': correlation})
    print(f"neut_weight: {neut_weight:.2f}, Pearson Correlation: {correlation:.4f}")

# Find the best neut_weight based on the highest Pearson correlation
optimal_result = max(correlation_results, key=lambda x: x['correlation'])
optimal_neut_weight = optimal_result['neut_weight']
optimal_correlation = optimal_result['correlation']

print(f"\nOptimal neut_weight: {optimal_neut_weight:.2f} with Pearson Correlation: {optimal_correlation:.4f}")
