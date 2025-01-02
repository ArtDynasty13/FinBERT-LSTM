import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Function to get valid date input
def get_date_input(prompt):
    while True:
        try:
            date_input = input(prompt)
            date = pd.to_datetime(date_input, format='%Y-%m-%d')
            return date
        except ValueError:
            print("Invalid date format. Please enter the date in YYYY-MM-DD format.")

# Load the aggregated sentiment data
file_path = "./91293_aggregated_sentiment.csv"
data = pd.read_csv(file_path)

# Convert 'PubDate' to datetime if it's not already
data['PubDate'] = pd.to_datetime(data['PubDate'])

# Ask the user for the date range
start_date = get_date_input("Enter the start date (YYYY-MM-DD): ")
end_date = get_date_input("Enter the end date (YYYY-MM-DD): ")

# Filter data based on the user input
data_filtered = data[(data['PubDate'] >= start_date) & (data['PubDate'] <= end_date)]

# Check if data exists for the given range
if data_filtered.empty:
    print("No data available for the given date range.")
else:
    # Apply a Savitzky-Golay filter to smooth the sentiment data (like a stock price)
    smoothed_sentiment = savgol_filter(data_filtered['average_sentiment'], window_length=11, polyorder=2)

    # Plotting the smoothed sentiment for the given date range
    plt.figure(figsize=(10, 6))
    plt.plot(data_filtered['PubDate'], smoothed_sentiment, color='b', label=f'Smoothed Sentiment ({start_date.date()} to {end_date.date()})', linewidth=2)

    # Add titles and labels
    plt.title(f'Smoothed Aggregated Sentiment from {start_date.date()} to {end_date.date()}')
    plt.xlabel('Date')
    plt.ylabel('Smoothed Average Sentiment')
    plt.grid(True)
    plt.xticks(rotation=45)

    # Show the plot with a legend
    plt.tight_layout()
    plt.legend()
    plt.show()
