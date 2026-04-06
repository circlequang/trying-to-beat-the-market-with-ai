import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
import os

# ==========================================
# Configuration Parameters
# ==========================================
FILE_PATH = '../data/BTCUSD4H.csv'
SPLIT_DATE = '2026-03-01'
SEQ_LENGTH = 30 # Number of previous time steps used to predict the next value

def create_sequences(data, seq_length):
    """
    Converts a time series array into sequences of length `seq_length`.
    """
    X = []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
    return np.array(X)

def main():
    # 1. Load the Dataset
    print(f"Loading data from {FILE_PATH}...")
    if not os.path.exists(FILE_PATH):
        print(f"Error: Could not find '{FILE_PATH}'. Please ensure the file exists.")
        return

    df = pd.read_csv(FILE_PATH)
    
    # Ensure datetime format and chronological order
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    # Define features. 
    # Note: Added 'low' to support the Pass/NG evaluation and chart.
    features = ['close', 'high', 'low', 'open', 'volume']
    
    # Check if all features exist in the dataframe
    missing_cols = [col for col in features if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns in CSV: {missing_cols}")
        return

    # 2. Split Data by Date Constraint
    # Train set: Before 2026.03.01
    # Test set: From 2026.03.01 to present
    train_df = df[df['datetime'] < SPLIT_DATE].copy()
    test_df = df[df['datetime'] >= SPLIT_DATE].copy()

    if train_df.empty or test_df.empty:
        print("Error: The dataset does not contain enough data to split by 2026-03-01.")
        return

    # 3. Data Scaling (Normalization)
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Fit the scaler ONLY on the training data to prevent data leakage
    train_scaled = scaler.fit_transform(train_df[features])
    
    # Prepend the last SEQ_LENGTH records from training set to test set
    combined_test_df = pd.concat([train_df.iloc[-SEQ_LENGTH:], test_df])
    test_scaled = scaler.transform(combined_test_df[features])

    # 4. Generate Input Sequences
    X_train = create_sequences(train_scaled, SEQ_LENGTH)
    # Target is the 'close' price (index 0) at the next time step
    y_train = train_scaled[SEQ_LENGTH:, 0] 

    X_test = create_sequences(test_scaled, SEQ_LENGTH)

    # 5. Build the RNN Architecture
    print("Building and compiling the RNN model...")
    model = Sequential([
        SimpleRNN(units=120, activation='relu', input_shape=(SEQ_LENGTH, len(features))),
        Dense(units=1) # Predicting the close price
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # 6. Train the Model
    print("Training the model (this may take a moment)...")
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)
    print("Training completed successfully.")

    # 7. Generate Predictions on the Test Set
    print("Generating predictions...")
    predicted_scaled = model.predict(X_test, verbose=0)

    # 8. Inverse Transform Predictions
    dummy_pred = np.zeros((len(predicted_scaled), len(features)))
    dummy_pred[:, 0] = predicted_scaled[:, 0]
    predicted_prices = scaler.inverse_transform(dummy_pred)[:, 0]

    # 9. Extract Actual Values for Evaluation directly from the DataFrame
    # This avoids floating point inaccuracies from scaling/unscaling actual data
    actual_test_data = combined_test_df.iloc[SEQ_LENGTH:].reset_index(drop=True)
    
    dates = actual_test_data['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S').values
    actual_close = actual_test_data['close'].values
    actual_high = actual_test_data['high'].values
    actual_low = actual_test_data['low'].values

    # 10. Calculate Previous Close Prices for Trend Evaluation
    # The previous close for the first test day is the last close of the training set.
    # For subsequent days, it is the previous day's actual close in the test set.
    last_train_close = combined_test_df.iloc[SEQ_LENGTH - 1]['close']
    prev_close_list = [last_train_close]
    if len(actual_close) > 1:
        prev_close_list.extend(actual_close[:-1])
    prev_actual_close = np.array(prev_close_list)

    # 11. Output to Console with Expanded Evaluations
    header = f"{'DATETIME':<20} | {'PREDICTED':<11} | {'ACTUAL CLS':<11} | {'HIGH':<11} | {'LOW':<11} | {'EVAL':<5} | {'TREND':<5} | {'PRED TREND':<10} | {'TRND EVAL'}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    
    for d, pred, close, high, low, prev_c in zip(dates, predicted_prices, actual_close, actual_high, actual_low, prev_actual_close):
        # Evaluation Rule 1: High-Low Range
        range_status = "Pass" if low <= pred <= high else "NG"
        
        # Evaluation Rule 2: Trend Evaluation (Compared to previous day's actual close)
        actual_trend = "UP" if close >= prev_c else "DOWN"
        pred_trend = "UP" if pred >= prev_c else "DOWN"
        trend_status = "Pass" if actual_trend == pred_trend else "NG"
        
        print(f"{d:<20} | {pred:<11.4f} | {close:<11.4f} | {high:<11.4f} | {low:<11.4f} | {range_status:<5} | {actual_trend:<5} | {pred_trend:<10} | {trend_status}")
    
    print("=" * len(header))

    # 12. Plotting the Chart
    print("Rendering chart...")
    plt.figure(figsize=(14, 7))
    
    # Plotting lines
    plt.plot(actual_test_data['datetime'], actual_close, label='Actual Close', color='blue', marker='o', markersize=4)
    plt.plot(actual_test_data['datetime'], predicted_prices, label='Predicted Close', color='orange', marker='x', markersize=4)
    plt.plot(actual_test_data['datetime'], actual_high, label='Actual High', color='green', linestyle='--', alpha=0.5)
    plt.plot(actual_test_data['datetime'], actual_low, label='Actual Low', color='red', linestyle='--', alpha=0.5)
    
    # Shading the High-Low area for better visual understanding
    plt.fill_between(actual_test_data['datetime'], actual_low, actual_high, color='gray', alpha=0.1, label='High-Low Range')

    # Formatting the chart
    plt.title('RNN Prediction vs Actual Prices (Test Set >= 2026-03-01)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
    main()