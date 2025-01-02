import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

def granger_causality_test():

    # Step 1: Load the cleaned data
    granger_data_df = pd.read_csv('granger_data.csv', parse_dates=['Date'], index_col='Date')

    # Step 2: Check for any remaining NaN values
    if granger_data_df.isna().sum().any():
        print("Warning: There are still NaN values in the dataset. Please check for missing data.")
    else:
        print("No NaN values found in the dataset.")

    # Step 3: Perform Granger Causality Test (Sentiment → Stock Price)
    print("\nGranger Causality Test: Sentiment → Stock Price")
    gc_result_sentiment_to_stock = grangercausalitytests(granger_data_df, maxlag=4, verbose=True)

    # Step 4: Perform Granger Causality Test (Stock Price → Sentiment)
    print("\nGranger Causality Test: Stock Price → Sentiment")
    gc_result_stock_to_sentiment = grangercausalitytests(granger_data_df[['Close', 'New_Sentiment']], maxlag=4, verbose=True)

if __name__ == '__main__':
    granger_causality_test()
