import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Function to preprocess data
def preprocess_data(merged_data, window_size, use_sentiment):
    if merged_data.empty:
        print("No data available after filtering. Please check the date range and data sources.")
        return None, None, None

    features = ['Close', 'New_Sentiment'] if use_sentiment else ['Close']
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(merged_data[features])
    merged_data[features] = normalized_data

    X, y = [], []
    for i in range(window_size, len(merged_data)):
        X.append(normalized_data[i-window_size:i])
        y.append(normalized_data[i, 0])
    
    return np.array(X), np.array(y), scaler

# Function to build and train the LSTM model
def build_and_train_lstm(X_train, y_train, X_val, y_val, units=50, dropout_rate=0.2, learning_rate=0.001, epochs=50, batch_size=32):
    from tensorflow.keras.optimizers import Adam

    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(dropout_rate),
        LSTM(units),
        Dropout(dropout_rate),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=1
    )
    return model, history

def evaluate_model(y_true, y_pred, scaler, sentiment=False):
    if sentiment:
        y_pred_rescaled = np.array([scaler.inverse_transform([[pred[0], 0]])[0, 0] for pred in y_pred])
        y_true_rescaled = np.array([scaler.inverse_transform([[true, 0]])[0, 0] for true in y_true])
    else:
        y_pred_rescaled = np.array([scaler.inverse_transform([[pred[0]]])[0, 0] for pred in y_pred])
        y_true_rescaled = np.array([scaler.inverse_transform([[true]])[0, 0] for true in y_true])
    return y_true_rescaled, y_pred_rescaled

if __name__ == '__main__':
    # Load data
    stock_data = pd.read_csv('./data/HXE.csv')
    sentiment_data = pd.read_csv('./data/91293_sentiment_model.csv')

    # Convert date columns to datetime format
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    sentiment_data['Date'] = pd.to_datetime(sentiment_data['Date'])

    # Merge datasets on 'Date'
    merged_data = pd.merge(stock_data, sentiment_data, on='Date', how='outer')

    # Sort by date and handle missing values
    merged_data.sort_values(by='Date', inplace=True)
    merged_data.fillna(method='ffill', inplace=True)
    merged_data.fillna(method='bfill', inplace=True)

    # Filter data within the desired date range
    start_date = datetime.strptime('2018-01-01', '%Y-%m-%d')
    end_date = datetime.strptime('2022-12-31', '%Y-%m-%d')
    merged_data = merged_data[(merged_data['Date'] >= start_date) & (merged_data['Date'] <= end_date)]

    # Check if data is available
    if not merged_data.empty:
        window_size = 1  # Set window size

        # Preprocess data with and without sentiment
        X_with_sentiment, y_with_sentiment, scaler_with_sentiment = preprocess_data(merged_data.copy(), window_size, use_sentiment=True)
        X_without_sentiment, y_without_sentiment, scaler_without_sentiment = preprocess_data(merged_data.copy(), window_size, use_sentiment=False)

        num_runs = 30  # Number of runs for averaging
        r2_ws_list, mse_ws_list, mae_ws_list = [], [], []  # Lists to store results for with sentiment
        r2_ns_list, mse_ns_list, mae_ns_list = [], [], []  # Lists to store results for without sentiment

        # Run the training and evaluation multiple times for averaging
        for run in range(num_runs):
            # Split data into training and testing sets
            X_train_ws, X_test_ws, y_train_ws, y_test_ws = train_test_split(X_with_sentiment, y_with_sentiment, test_size=0.2, shuffle=False)
            X_train_ns, X_test_ns, y_train_ns, y_test_ns = train_test_split(X_without_sentiment, y_without_sentiment, test_size=0.2, shuffle=False)

            # Train models
            model_ws, _ = build_and_train_lstm(X_train_ws, y_train_ws, X_test_ws, y_test_ws)
            model_ns, _ = build_and_train_lstm(X_train_ns, y_train_ns, X_test_ns, y_test_ns)

            # Get predictions
            predictions_ws = model_ws.predict(X_test_ws)
            predictions_ns = model_ns.predict(X_test_ns)

            # Evaluate "With Sentiment" model
            y_test_actual_ws, predictions_ws_rescaled = evaluate_model(y_test_ws, predictions_ws, scaler_with_sentiment, sentiment=True)
            mse_ws = mean_squared_error(y_test_actual_ws, predictions_ws_rescaled)
            mae_ws = mean_absolute_error(y_test_actual_ws, predictions_ws_rescaled)
            r2_ws_list.append(r2_score(y_test_actual_ws, predictions_ws_rescaled))
            mse_ws_list.append(mse_ws)
            mae_ws_list.append(mae_ws)

            # Evaluate "Without Sentiment" model
            y_test_actual_ns, predictions_ns_rescaled = evaluate_model(y_test_ns, predictions_ns, scaler_without_sentiment, sentiment=False)
            mse_ns = mean_squared_error(y_test_actual_ns, predictions_ns_rescaled)
            mae_ns = mean_absolute_error(y_test_actual_ns, predictions_ns_rescaled)
            r2_ns_list.append(r2_score(y_test_actual_ns, predictions_ns_rescaled))
            mse_ns_list.append(mse_ns)
            mae_ns_list.append(mae_ns)

        # Calculate average metrics
        avg_r2_ws = np.mean(r2_ws_list)
        avg_mse_ws = np.mean(mse_ws_list)
        avg_mae_ws = np.mean(mae_ws_list)

        avg_r2_ns = np.mean(r2_ns_list)
        avg_mse_ns = np.mean(mse_ns_list)
        avg_mae_ns = np.mean(mae_ns_list)

        # Print the results
        print(f"\nAverage Metrics Over {num_runs} Run(s):")
        print(f"With Sentiment - R²: {avg_r2_ws:.4f}, MSE: {avg_mse_ws:.4f}, MAE: {avg_mae_ws:.4f}")
        print(f"Without Sentiment - R²: {avg_r2_ns:.4f}, MSE: {avg_mse_ns:.4f}, MAE: {avg_mae_ns:.4f}")