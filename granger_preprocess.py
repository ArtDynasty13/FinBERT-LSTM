import pandas as pd
import os

def granger_preprocess_data(start_date=None, end_date=None):
    # Define the path to the 'data' folder
    data_folder = 'data'
    
    # Step 1: Load the datasets from the 'data' folder
    sentiment_df = pd.read_csv(os.path.join(data_folder, '91293_sentiment_model.csv'), parse_dates=['Date'], index_col='Date')
    stock_df = pd.read_csv(os.path.join(data_folder, 'HXE.csv'), parse_dates=['Date'], index_col='Date')

    # Step 2: Determine default start and end dates if not provided
    sentiment_start_date = sentiment_df.index.min()
    stock_start_date = pd.to_datetime('2013-09-17')

    # Default start and end dates
    if start_date is None:
        start_date = max(sentiment_start_date, stock_start_date)
    if end_date is None:
        end_date = sentiment_df.index.max()

    # Ensure start_date and end_date are in datetime format
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Step 3: Merge the two datasets and align on the Date index
    merged_df = pd.merge(sentiment_df[['New_Sentiment']], stock_df[['Close']], left_index=True, right_index=True, how='outer')

    # Step 4: Filter rows within the specified date range
    merged_df = merged_df[(merged_df.index >= start_date) & (merged_df.index <= end_date)]

    # Step 5: Forward fill missing values
    merged_df.ffill(inplace=True)

    # Step 6: Back fill remaining NaN values, if any
    if merged_df.isna().sum().any():
        print("NaN values found after forward filling. Applying back fill.")
        merged_df.bfill(inplace=True)

    # Step 7: Check for any remaining NaN values after back filling
    print("Number of NaN values remaining in the dataset after back-fill:")
    print(merged_df.isna().sum())

    # Step 8: Save the cleaned dataset to the 'data' folder
    merged_df.to_csv(os.path.join(data_folder, 'granger_data.csv'))
    print("Cleaned data saved to 'data/granger_data.csv'")

    print(f"Executing Granger Causality test from {start_date} to {end_date}...")

if __name__ == '__main__':
    granger_preprocess_data()
