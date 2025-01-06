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
    # Check if the merged data is empty after filtering
    if merged_data.empty:
        print("No data available after filtering. Please check the date range and data sources.")
        return None, None, None

    # Select features based on whether sentiment is included or not
    if use_sentiment:
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'New_Sentiment']
    else:
        features = ['Open', 'High', 'Low', 'Close', 'Volume']

    # Normalize the data
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(merged_data[features])
    merged_data[features] = normalized_data

    # Create input windows and labels
    X, y = [], []
    for i in range(window_size, len(merged_data)):
        X.append(normalized_data[i-window_size:i])  # Input features for window_size days
        y.append(normalized_data[i, 3])  # Target is the normalized 'Close' price
    
    return np.array(X), np.array(y), scaler

# Function to build and train the LSTM model
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

# Function to evaluate the model
def evaluate_model(y_true, y_pred, scaler, sentiment=False):
    # Reverse normalization
    if sentiment:
        y_pred_rescaled = scaler.inverse_transform([[0, 0, 0, pred[0], 0, 0] for pred in y_pred])[:, 3]
        y_true_rescaled = scaler.inverse_transform([[0, 0, 0, true, 0, 0] for true in y_true])[:, 3]
    else:
        y_pred_rescaled = scaler.inverse_transform([[0, 0, 0, pred[0], 0] for pred in y_pred])[:, 3]
        y_true_rescaled = scaler.inverse_transform([[0, 0, 0, true, 0] for true in y_true])[:, 3]

    # Calculate R², MSE, and MAE
    r2 = r2_score(y_true_rescaled, y_pred_rescaled)
    mse = mean_squared_error(y_true_rescaled, y_pred_rescaled)
    mae = mean_absolute_error(y_true_rescaled, y_pred_rescaled)

    # Print out the metrics
    print(f"Metrics for {'Sentiment' if sentiment else 'Non-Sentiment'} Model:")
    print(f"R²: {r2:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    
    return y_true_rescaled, y_pred_rescaled

# Function to plot results
def plot_results(test_dates, y_true_ws, y_pred_ws, y_true_ns, y_pred_ns):
    plt.figure(figsize=(14,7))
    
    # Plot for Sentiment Model
    plt.subplot(1, 2, 1)
    plt.plot(test_dates, y_true_ws, label='Actual Price', color='blue')
    plt.plot(test_dates, y_pred_ws, label='Predicted Price', color='red')
    plt.title('Sentiment Model: Actual vs Predicted')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    
    # Plot for Non-Sentiment Model
    plt.subplot(1, 2, 2)
    plt.plot(test_dates, y_true_ns, label='Actual Price', color='blue')
    plt.plot(test_dates, y_pred_ns, label='Predicted Price', color='red')
    plt.title('Non-Sentiment Model: Actual vs Predicted')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    
    # Show plot
    plt.tight_layout()
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

    # Convert 'start_date' and 'end_date' to datetime objects
    start_date = datetime.strptime('2014-01-01', '%Y-%m-%d')  # Updated start date
    end_date = datetime.strptime('2018-12-31', '%Y-%m-%d')

    # Filter the merged data based on the date range
    merged_data = merged_data[(merged_data['Date'] >= start_date) & (merged_data['Date'] <= end_date)]
    print(f"Data range after merge: {merged_data['Date'].min()} to {merged_data['Date'].max()}")

    # Check if the data is empty after filtering
    if merged_data.empty:
        print("No data available after filtering. Please check the date range and data sources.")
    else:
        print(f"Filtered data contains {len(merged_data)} rows.")
        print(f"Data range after filtering: {merged_data['Date'].min()} to {merged_data['Date'].max()}")

        # Preprocess data for both scenarios
        window_size = 1  # Experiment with other sizes
        
        # With sentiment
        X_with_sentiment, y_with_sentiment, scaler_with_sentiment = preprocess_data(merged_data.copy(), window_size, use_sentiment=True)
        
        # Without sentiment
        X_without_sentiment, y_without_sentiment, scaler_without_sentiment = preprocess_data(merged_data.copy(), window_size, use_sentiment=False)

        if X_with_sentiment is None or y_with_sentiment is None or X_without_sentiment is None or y_without_sentiment is None:
            print("Skipping model training due to empty data.")
        else:
            # Train-test split for both scenarios
            X_train_ws, X_test_ws, y_train_ws, y_test_ws = train_test_split(X_with_sentiment, y_with_sentiment, test_size=0.2, shuffle=False)
            X_train_ws, X_val_ws, y_train_ws, y_val_ws = train_test_split(X_train_ws, y_train_ws, test_size=0.2, shuffle=False)

            X_train_ns, X_test_ns, y_train_ns, y_test_ns = train_test_split(X_without_sentiment, y_without_sentiment, test_size=0.2, shuffle=False)
            X_train_ns, X_val_ns, y_train_ns, y_val_ns = train_test_split(X_train_ns, y_train_ns, test_size=0.2, shuffle=False)

            # Build and train models for both scenarios
            model_ws, _ = build_and_train_lstm(X_train_ws, y_train_ws, X_val_ws, y_val_ws, units=50, dropout_rate=0.2, epochs=50, batch_size=32)
            model_ns, _ = build_and_train_lstm(X_train_ns, y_train_ns, X_val_ns, y_val_ns, units=50, dropout_rate=0.2, epochs=50, batch_size=32)

            # Predictions for both scenarios
            predictions_ws = model_ws.predict(X_test_ws)
            predictions_ns = model_ns.predict(X_test_ns)

            # Evaluate the models
            y_test_actual_ws, predictions_ws = evaluate_model(y_test_ws, predictions_ws, scaler_with_sentiment, sentiment=True)
            y_test_actual_ns, predictions_ns = evaluate_model(y_test_ns, predictions_ns, scaler_without_sentiment, sentiment=False)

            # Align dates with test data
            test_dates = merged_data['Date'].iloc[-len(y_test_actual_ws):].values

            # Plot results
            plot_results(test_dates, y_test_actual_ws, predictions_ws, y_test_actual_ns, predictions_ns)

            # Save results to CSV
            results = pd.DataFrame({
                'Date': test_dates,
                'Actual Price': y_test_actual_ws,
                'Predicted Price (With Sentiment)': predictions_ws,
                'Predicted Price (Without Sentiment)': predictions_ns
            })
            results.to_csv('./data/predictions.csv', index=False)
            print("Predictions saved to './data/predictions.csv'")
