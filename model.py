import pandas as pd
from model_plot import plot_sentiment_stock
from granger_preprocess import granger_preprocess_data
from granger_test import granger_causality_test

# Load the data
file_path = "./data/91293_sentiment.csv"
data = pd.read_csv(file_path)

# Convert 'PubDate' to datetime
data['PubDate'] = pd.to_datetime(data['PubDate'])

# Ask user if they want to specify a custom date range for sentiment
custom_range = input("Would you like to select a custom date range? (y/n, default is 'n'): ").strip().lower()

start_date = '2014-01-01'
end_date = '2024-12-31'

# Default to 'n' if input is empty
if custom_range == '':
    custom_range = 'n'

if custom_range == 'y':
    start_date = input("Enter the start date (YYYY-MM-DD): ").strip()
    end_date = input("Enter the end date (YYYY-MM-DD): ").strip()
# Convert string inputs to datetime.date
start_date = pd.to_datetime(start_date).date()
end_date = pd.to_datetime(end_date).date()

# Filter the data based on the user input
data = data[(data['PubDate'].dt.date >= start_date) & (data['PubDate'].dt.date <= end_date)]

# Map sentiment labels to numerical values (-1, 0, 1)
sentiment_mapping = {'positive': 1, 'neutral': 0, 'negative': -1}
data['sentiment_value'] = data['sentiment'].map(sentiment_mapping)

# Initialize variables for sentiment calculations
nudge_strength = 0.1
neut_weight = 0.2
current_sentiment = 0  # Start with an initial sentiment value of 0

# Create a list to store daily sentiment data
daily_sentiment_data = []

# Iterate over the filtered data and calculate the daily sentiment dynamics
for date, group in data.groupby('PubDate'):
    # Count positive, neutral, and negative sentiment for the day
    pos_count = group[group['sentiment_value'] == 1].shape[0]
    neut_count = group[group['sentiment_value'] == 0].shape[0]
    neg_count = group[group['sentiment_value'] == -1].shape[0]
    
    # Calculate the new sentiment based on the previous sentiment
    new_sentiment = current_sentiment  # Start with the current sentiment

    # Adjust sentiment based on the counts
    new_sentiment += pos_count * nudge_strength  # Positive articles nudge the sentiment up
    new_sentiment -= neg_count * nudge_strength  # Negative articles nudge the sentiment down
    new_sentiment += neut_count * nudge_strength * -neut_weight if new_sentiment > 0 else neut_count * nudge_strength * neut_weight

    # Store the results for the day
    daily_sentiment_data.append({
        'Date': date,
        'Pos_Count': pos_count,
        'Neut_Count': neut_count,
        'Neg_Count': neg_count,
        'New_Sentiment': new_sentiment
    })

    # Update the current sentiment for the next day
    current_sentiment = new_sentiment

# Convert the list of daily sentiment data into a DataFrame
daily_sentiment_df = pd.DataFrame(daily_sentiment_data)

# Save the results to a new CSV file
output_file = "./data/91293_sentiment_model.csv"
daily_sentiment_df.to_csv(output_file, index=False)

# Run Granger Test
#granger_preprocess_data()
#granger_causality_test()

# Now that the filtered CSV is generated, pass the custom date range (or entire range) to the plot function
if custom_range == 'y':
    plot_sentiment_stock(start_date, end_date)
else:
    plot_sentiment_stock()