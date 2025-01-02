import pandas as pd

# Load the daily sentiment data
file_path = "./91293_sentiment_model.csv"  # Update this path if needed
data = pd.read_csv(file_path)

# Convert 'Date' to datetime
if 'Date' in data.columns:
    data['Date'] = pd.to_datetime(data['Date'])

# Ensure the necessary columns exist
if not {'Pos_Count', 'Neg_Count'}.issubset(data.columns):
    raise ValueError("The CSV file must contain 'Positive' and 'Negative' columns.")

# Calculate the sentiment index (BI) based on the formula: BI = (M^P - M^N) / (M^P + M^N)
def calculate_sentiment_index(row):
    positive = row['Pos_Count']
    negative = row['Neg_Count']
    if positive + negative == 0:
        return 0  # Handle division by zero
    return (positive - negative) / (positive + negative)

# Apply the formula to calculate BI
data['Sentiment_Index'] = data.apply(calculate_sentiment_index, axis=1)

# Save the updated data to a new CSV file
output_file = "./91293_paper_sentiment_model.csv"
data.to_csv(output_file, index=False)

print(f"Sentiment index calculated and saved to {output_file}.")