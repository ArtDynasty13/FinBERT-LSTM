import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime

# Function to preprocess data
def preprocess_data(merged_data, window_size):
    # Check if the merged data is empty after filtering
    if merged_data.empty:
        print("No data available after filtering. Please check the date range and data sources.")
        return None, None, None

    # Select features and normalize
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'New_Sentiment']
    scaler = MinMaxScaler()
    
    # Ensure we have data for normalization
    if len(merged_data[features]) > 0:
        normalized_data = scaler.fit_transform(merged_data[features])
        merged_data[features] = normalized_data
    else:
        raise ValueError("No data available for normalization.")

    # Create input windows and labels
    X, y = [], []
    for i in range(window_size, len(merged_data)):
        X.append(normalized_data[i-window_size:i])  # Input features for window_size days
        y.append(normalized_data[i, 3])  # Target is the normalized 'Close' price
    
    return np.array(X), np.array(y), scaler


def build_and_train_lstm(X_train, y_train, X_val, y_val, units=50, dropout_rate=0.2, learning_rate=0.001, epochs=50, batch_size=32):
    from tensorflow.keras.optimizers import Adam

    model = Sequential([ 
        LSTM(units, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(dropout_rate),
        LSTM(units),
        Dropout(dropout_rate),
        Dense(1)  # Output layer for regression
    ])
    
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=1
    )
    
    return model, history

def plot_predictions(model, X_test, y_test, scaler, original_data):
    # Make predictions
    predictions = model.predict(X_test)
    
    # Reverse normalization for comparison
    predictions = scaler.inverse_transform([[0, 0, 0, pred[0], 0, 0] for pred in predictions])[:, 3]
    y_test_actual = scaler.inverse_transform([[0, 0, 0, true, 0, 0] for true in y_test])[:, 3]
    
    # Plot actual vs predicted
    plt.figure(figsize=(12, 6))
    plt.plot(original_data['Date'][-len(y_test_actual):], y_test_actual, label="Actual Price", color='blue')
    plt.plot(original_data['Date'][-len(predictions):], predictions, label="Predicted Price", color='red')
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.title("Actual vs Predicted Stock Prices")
    plt.show()

if __name__ == '__main__':

    # Load stock and sentiment data
    stock_data = pd.read_csv('./data/HXE.csv')  # Adjust path if needed
    sentiment_data = pd.read_csv('./data/91293_sentiment_model.csv')  # Adjust path if needed

    # Convert 'Date' to datetime format
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    sentiment_data['Date'] = pd.to_datetime(sentiment_data['Date'])

    # Merge data on the 'Date' column
    merged_data = pd.merge(stock_data, sentiment_data, on='Date', how='outer')

    # Handle missing values
    merged_data.sort_values(by='Date', inplace=True)  # Sort by Date
    merged_data.fillna(method='ffill', inplace=True)  # Forward fill missing values
    merged_data.fillna(method='bfill', inplace=True)  # Backward fill for remaining gaps

    # Save the merged data to a CSV file
    merged_data.to_csv('./data/merged.csv', index=False)

    print("Merged data saved to './data/merged.csv'")

    # Print the first few rows to check the merge
    print(merged_data.head())

    # Convert 'start_date' and 'end_date' to datetime objects
    start_date = datetime.strptime('2014-01-01', '%Y-%m-%d')  # Updated start date
    end_date = datetime.strptime('2018-12-31', '%Y-%m-%d')

    # Filter the merged data based on the date range
    merged_data = merged_data[(merged_data['Date'] >= start_date) & (merged_data['Date'] <= end_date)]

    # Check if the data is empty after filtering
    if merged_data.empty:
        print("No data available after filtering. Please check the date range and data sources.")
    else:
        print(f"Filtered data contains {len(merged_data)} rows.")

        # Preprocess data
        window_size = 10  # Experiment with other sizes
        X, y, scaler = preprocess_data(merged_data, window_size)

        if X is None or y is None:
            print("Skipping model training due to empty data.")
        else:
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)

            # Build and train model
            model, history = build_and_train_lstm(X_train, y_train, X_val, y_val, units=50, dropout_rate=0.2, epochs=50, batch_size=32)

            # Plot predictions
            plot_predictions(model, X_test, y_test, scaler, stock_data)
