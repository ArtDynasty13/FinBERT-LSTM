import pandas as pd

# Load the sentiment analysis results CSV
file_path = "./91293_sentiment.csv"
data = pd.read_csv(file_path)

# Map sentiment labels to numerical values (-1, 0, 1)
sentiment_mapping = {'positive': 1, 'neutral': 0, 'negative': -1}

# Apply the mapping to create the sentiment_value column
data['sentiment_value'] = data['sentiment'].map(sentiment_mapping)

# Check for rows where the sentiment_value is NaN (unmapped sentiments)
print(data[data['sentiment_value'].isna()])

# Optionally, handle any rows with NaN sentiment_value (e.g., drop or fix them)
# For now, we'll drop rows with NaN values in sentiment_value
data = data.dropna(subset=['sentiment_value'])

# Show the first few rows to verify the result
print(data[['PubDate', 'sentiment', 'sentiment_value']].head())

# Aggregate the data by PubDate and calculate the average sentiment
# We are grouping by 'PubDate' and calculating the mean of 'sentiment_value'
daily_sentiment = data.groupby('PubDate').agg(
    average_sentiment=('sentiment_value', 'mean')
).reset_index()

# Display the aggregated result
print(daily_sentiment.head())

# Optionally, save the aggregated data to a new CSV file
output_file = "./91293_aggregated_sentiment.csv"
daily_sentiment.to_csv(output_file, index=False)
print(f"Results saved to {output_file}")
