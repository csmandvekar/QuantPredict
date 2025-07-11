import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
import os
warnings.filterwarnings('ignore')

class CNNLSTMStockPredictor:
    def __init__(self, sequence_length=60, future_days=30):
        self.sequence_length = sequence_length
        self.future_days = future_days
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.feature_columns = ['open', 'high', 'low', 'close', 'volume']
        
    def create_technical_indicators(self, df):
        """Create technical indicators to enhance the dataset"""
        df = df.copy()
        
        # Moving averages
        df['ma_7'] = df['close'].rolling(window=7).mean()
        df['ma_21'] = df['close'].rolling(window=21).mean()
        df['ma_50'] = df['close'].rolling(window=50).mean()
        
        # RSI (Relative Strength Index)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # Price change percentage
        df['price_change'] = df['close'].pct_change()
        
        # Volume moving average
        df['volume_ma'] = df['volume'].rolling(window=10).mean()
        
        # High-Low percentage
        df['hl_pct'] = (df['high'] - df['low']) / df['close'] * 100
        
        # Close-Open percentage
        df['co_pct'] = (df['close'] - df['open']) / df['open'] * 100
        
        return df
    
    def standardize_column_names(self, df):
        """Standardize column names to lowercase and handle different formats"""
        # Create a copy to avoid modifying original
        df = df.copy()
        
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join([str(level) for level in col if str(level) != '']) for col in df.columns.values]
        
        # Convert all column names to lowercase and remove spaces
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        
        # Handle different possible column name formats
        column_mapping = {}
        
        # Check for date columns
        date_columns = ['date', 'timestamp', 'time', 'dt']
        for col in df.columns:
            if any(date_col in col.lower() for date_col in date_columns):
                column_mapping[col] = 'date'
                break
        
        # Check for OHLCV columns, including those with suffixes (e.g., open_infy.ns)
        price_mappings = {
            'open': ['open'],
            'high': ['high'],
            'low': ['low'],
            'close': ['close', 'adj_close', 'close_price'],
            'volume': ['volume']
        }
        for standard_name, possible_names in price_mappings.items():
            for col in df.columns:
                col_lower = col.lower()
                # Map columns that start with the price name (e.g., open, open_infy.ns)
                if any(col_lower == name or col_lower.startswith(f"{name}_") for name in possible_names):
                    column_mapping[col] = standard_name
        
        # Apply the mapping
        df = df.rename(columns=column_mapping)
        
        return df
    
    def prepare_data(self, df):
        """Prepare and preprocess the data"""
        # Standardize column names
        df = self.standardize_column_names(df)
        
        print(f"Standardized column names: {list(df.columns)}")
        
        # Check if we have all required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            print(f"Available columns: {list(df.columns)}")
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Convert date column to datetime if it exists
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        else:
            # If no date column, create one based on index
            print("No date column found, creating date index...")
            df['date'] = pd.date_range(start='2019-01-01', periods=len(df), freq='D')
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        # Create technical indicators
        df = self.create_technical_indicators(df)
        
        # Select features for training
        self.feature_columns = ['open', 'high', 'low', 'close', 'volume', 
                               'ma_7', 'ma_21', 'ma_50', 'rsi', 'macd', 
                               'macd_signal', 'bb_upper', 'bb_lower', 
                               'price_change', 'volume_ma', 'hl_pct', 'co_pct']
        
        # Drop rows with NaN values
        df_clean = df.dropna().reset_index(drop=True)
        
        print(f"Original data shape: {df.shape}")
        print(f"Clean data shape after adding indicators: {df_clean.shape}")
        print(f"Data range: {df_clean['date'].min()} to {df_clean['date'].max()}")
        
        # Scale the features
        scaled_data = self.scaler.fit_transform(df_clean[self.feature_columns])
        
        return scaled_data, df_clean
    
    def create_sequences(self, data, target_column_index=3):  # close price is at index 3
        """Create sequences for training"""
        X, y = [], []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(data[i, target_column_index])  # Close price
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        """Build CNN-LSTM model architecture"""
        model = Sequential([
            # LSTM layers
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            
            # Reshape for CNN (this is a workaround since we need sequence for CNN)
            # We'll use Conv1D on the LSTM output features
            tf.keras.layers.RepeatVector(1),
            tf.keras.layers.Reshape((1, 32)),
            
            # CNN layers
            Conv1D(filters=64, kernel_size=1, activation='relu'),
            Conv1D(filters=32, kernel_size=1, activation='relu'),
            Flatten(),
            
            # Fully connected layers
            Dense(50, activation='relu'),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=10):
        """Train the CNN-LSTM model"""
        self.model = self.build_model((X_train.shape[1], X_train.shape[2]))
        
        print("Model Architecture:")
        self.model.summary()
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=0.0001,
            verbose=1
        )
        
        # Train the model
        print("\nStarting model training...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return history
    
    def predict_future(self, last_sequence, steps=30):
        """Predict future prices using sliding window approach"""
        predictions = []
        current_sequence = last_sequence.copy()
        
        print(f"\nPredicting next {steps} days...")
        
        for i in range(steps):
            # Predict next value
            next_pred = self.model.predict(current_sequence.reshape(1, *current_sequence.shape), verbose=0)
            predictions.append(next_pred[0, 0])
            
            # Update sequence: remove first element and add prediction
            new_row = current_sequence[-1].copy()
            new_row[3] = next_pred[0, 0]  # Update close price (index 3)
            
            # For other features, we'll use simple heuristics or carry forward
            # This is a simplification - in practice, you might want more sophisticated feature engineering
            current_sequence = np.vstack([current_sequence[1:], new_row.reshape(1, -1)])
            
            if (i + 1) % 10 == 0:
                print(f"Predicted {i + 1}/{steps} days")
        
        return np.array(predictions)
    
    def inverse_transform_predictions(self, predictions):
        """Convert scaled predictions back to original scale"""
        # Create a dummy array with the same shape as original features
        dummy = np.zeros((len(predictions), len(self.feature_columns)))
        dummy[:, 3] = predictions  # Close price is at index 3
        
        # Inverse transform
        inverse_scaled = self.scaler.inverse_transform(dummy)
        return inverse_scaled[:, 3]  # Return only close prices
    
    def evaluate_model(self, y_true, y_pred):
        """Evaluate model performance"""
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }
    
    def plot_training_history(self, history):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(history.history['loss'], label='Training Loss')
        ax1.plot(history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot MAE
        ax2.plot(history.history['mae'], label='Training MAE')
        ax2.plot(history.history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_predictions(self, dates, actual_prices, predicted_prices, title="Stock Price Prediction", future_dates=None, future_prices=None):
        """Plot actual vs predicted prices"""
        plt.figure(figsize=(15, 8))
        
        # Plot historical data
        plt.plot(dates, actual_prices, label='Actual Prices', color='blue', linewidth=2)
        plt.plot(dates, predicted_prices, label='Predicted Prices', color='red', linewidth=2, linestyle='--')
        
        # Plot future predictions if provided
        if future_dates is not None and future_prices is not None:
            plt.plot(future_dates, future_prices, label='Future Predictions', color='green', linewidth=2, linestyle=':')
        
        plt.title(title, fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Close Price', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def save_model(self, path):
        """Save the Keras model, scaler, and feature columns to disk."""
        # Save the Keras model
        model_path = path.replace('.pkl', '.h5')
        self.model.save(model_path)
        # Save the scaler
        import joblib
        scaler_path = path.replace('.pkl', '_scaler.save')
        joblib.dump(self.scaler, scaler_path)
        # Save the feature columns
        import json
        features_path = model_path.replace('.h5', '_features.json')
        with open(features_path, 'w') as f:
            json.dump(self.feature_columns, f)

# Main execution
def main():
    print("CNN-LSTM Stock Price Prediction Model for INFY")
    print("=" * 50)
    
    # Load the data
    data_path = "dataset/INFY.csv"
    
    if not os.path.exists(data_path):
        print(f"Error: File {data_path} not found!")
        print("Please make sure the file exists in the dataset folder.")
        return
    
    # Read the CSV file
    df = pd.read_csv(data_path)
    print(f"Data loaded successfully!")
    print(f"Columns in dataset: {list(df.columns)}")
    print(f"Dataset shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())
    
    # Initialize the predictor
    predictor = CNNLSTMStockPredictor(sequence_length=60, future_days=30)
    
    # Prepare data
    print("\n" + "="*50)
    print("DATA PREPROCESSING")
    print("="*50)
    scaled_data, df_processed = predictor.prepare_data(df)
    
    # Create sequences
    X, y = predictor.create_sequences(scaled_data)
    print(f"Sequences created - X shape: {X.shape}, y shape: {y.shape}")
    
    # Split data (80% train, 20% validation)
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    
    # Train model
    print("\n" + "="*50)
    print("MODEL TRAINING")
    print("="*50)
    history = predictor.train_model(X_train, y_train, X_val, y_val, epochs=100)
    
    # Plot training history
    predictor.plot_training_history(history)
    
    # Make predictions for validation set
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    val_predictions = predictor.model.predict(X_val)
    
    # Convert back to original scale
    val_predictions_scaled = predictor.inverse_transform_predictions(val_predictions.flatten())
    y_val_scaled = predictor.inverse_transform_predictions(y_val)
    
    # Evaluate model
    metrics = predictor.evaluate_model(y_val_scaled, val_predictions_scaled)
    print("Validation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Get validation dates for plotting
    val_start_idx = split_idx + predictor.sequence_length
    val_dates = df_processed['date'].iloc[val_start_idx:val_start_idx + len(y_val_scaled)]
    
    # Plot validation results
    predictor.plot_predictions(
        val_dates, 
        y_val_scaled, 
        val_predictions_scaled, 
        "INFY Stock Price - Validation Results"
    )
    
    # Predict future 30 days
    print("\n" + "="*50)
    print("FUTURE PREDICTIONS")
    print("="*50)
    last_sequence = X[-1]  # Last 60 days of data
    future_predictions = predictor.predict_future(last_sequence, steps=30)
    future_prices = predictor.inverse_transform_predictions(future_predictions)
    
    # Create future dates
    last_date = df_processed['date'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='D')
    
    print(f"\nPredicted INFY close prices for next 30 days:")
    print("-" * 40)
    for i, (date, price) in enumerate(zip(future_dates, future_prices), 1):
        print(f"Day {i:2d} ({date.strftime('%Y-%m-%d')}): ₹{price:.2f}")
    
    # Plot recent data with future predictions
    recent_data_points = 100  # Show last 100 days
    recent_start_idx = max(0, len(df_processed) - recent_data_points)
    recent_dates = df_processed['date'].iloc[recent_start_idx:]
    recent_prices = df_processed['close'].iloc[recent_start_idx:]
    
    predictor.plot_predictions(
        recent_dates,
        recent_prices,
        recent_prices,  # Actual prices for recent data
        "INFY Stock Price - Recent Data + Future Predictions",
        future_dates,
        future_prices
    )
    
    # Summary statistics
    print("\n" + "="*50)
    print("PREDICTION SUMMARY")
    print("="*50)
    current_price = df_processed['close'].iloc[-1]
    avg_future_price = np.mean(future_prices)
    price_change = ((avg_future_price - current_price) / current_price) * 100
    
    print(f"Current INFY Close Price: ₹{current_price:.2f}")
    print(f"Average Predicted Price (30 days): ₹{avg_future_price:.2f}")
    print(f"Expected Price Change: {price_change:+.2f}%")
    print(f"Price Range (30 days): ₹{np.min(future_prices):.2f} - ₹{np.max(future_prices):.2f}")
    
    # Save predictions to CSV
    predictions_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Close': future_prices
    })
    predictions_df.to_csv('future_predictions_INFY.csv', index=False)
    print(f"\nFuture predictions saved to 'future_predictions_INFY.csv'")

if __name__ == "__main__":
    main()