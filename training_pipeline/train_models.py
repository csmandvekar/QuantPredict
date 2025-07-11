#!/usr/bin/env python3
"""
Stock Price Prediction Model Training Pipeline
==============================================

This script trains CNN-LSTM models for stock price prediction using:
- Data from yfinance scraper
- CNNLSTMStockPredictor model architecture
- Automated training for multiple stocks
- Model saving to pretrained_models directory

Usage:
    python train_models.py --stocks INFY,TCS,RELIANCE --epochs 50
    python train_models.py --all --epochs 100
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add backend to path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from app.services.scraper import fetch_stock_data
from app.services.claude_model import CNNLSTMStockPredictor
from app.core.config import MODEL_DIR, DATA_DIR, ALLOWED_SCRIPTS

class ModelTrainer:
    def __init__(self, sequence_length=60, future_days=30):
        self.sequence_length = sequence_length
        self.future_days = future_days
        self.models = {}
        self.training_history = {}
        
    def fetch_and_prepare_data(self, script, start_date="2020-01-01"):
        """Fetch stock data and prepare it for training. Script should be passed without .NS extension."""
        print(f"\n📊 Fetching data for {script}...")
        try:
            # Fetch data using existing scraper
            df = fetch_stock_data(script, start_date)
            print(f"✅ Fetched {len(df)} records for {script}")
            print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")
            return df
        except Exception as e:
            print(f"❌ Error fetching data for {script}: {str(e)}")
            return None
    
    def train_model_for_script(self, script, epochs=50, validation_split=0.2):
        """Train a model for a specific stock script"""
        print(f"\n🚀 Training model for {script}...")
        
        # Fetch data
        df = self.fetch_and_prepare_data(script)
        if df is None:
            return False
        
        try:
            # Initialize model
            model = CNNLSTMStockPredictor(
                sequence_length=self.sequence_length,
                future_days=self.future_days
            )
            
            # Prepare data
            print(f"🔧 Preparing data for {script}...")
            scaled_data, df_clean = model.prepare_data(df)
            
            # Create sequences
            X, y = model.create_sequences(scaled_data)
            print(f"📈 Created {len(X)} training sequences")
            
            # Split data
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            print(f"📊 Training set: {len(X_train)} samples")
            print(f"📊 Validation set: {len(X_val)} samples")
            
            # Train model
            print(f"🎯 Training model for {epochs} epochs...")
            history = model.train_model(X_train, y_train, X_val, y_val, epochs=epochs)
            
            # Save model
            model_path = os.path.join(MODEL_DIR, f"{script}_model.pkl")
            model.save_model(model_path)
            print(f"💾 Model saved to: {model_path}")
            
            # Store model and history
            self.models[script] = model
            self.training_history[script] = history
            
            # Evaluate model
            y_pred = model.model.predict(X_val)
            mse = np.mean((y_val - y_pred.flatten()) ** 2)
            mae = np.mean(np.abs(y_val - y_pred.flatten()))
            
            print(f"📊 {script} Training Results:")
            print(f"   MSE: {mse:.6f}")
            print(f"   MAE: {mae:.6f}")
            print(f"   Final Loss: {history.history['loss'][-1]:.6f}")
            print(f"   Final Val Loss: {history.history['val_loss'][-1]:.6f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error training model for {script}: {str(e)}")
            return False
    
    def train_multiple_models(self, scripts, epochs=50):
        """Train models for multiple scripts"""
        print(f"🎯 Starting training for {len(scripts)} stocks...")
        print(f"📋 Stocks: {', '.join(scripts)}")
        
        successful_trains = []
        failed_trains = []
        
        for i, script in enumerate(scripts, 1):
            print(f"\n{'='*60}")
            print(f"📈 Training {i}/{len(scripts)}: {script}")
            print(f"{'='*60}")
            
            success = self.train_model_for_script(script, epochs=epochs)
            
            if success:
                successful_trains.append(script)
            else:
                failed_trains.append(script)
        
        # Summary
        print(f"\n{'='*60}")
        print(f"🎉 TRAINING SUMMARY")
        print(f"{'='*60}")
        print(f"✅ Successful: {len(successful_trains)}")
        print(f"❌ Failed: {len(failed_trains)}")
        
        if successful_trains:
            print(f"✅ Trained models for: {', '.join(successful_trains)}")
        
        if failed_trains:
            print(f"❌ Failed to train: {', '.join(failed_trains)}")
        
        return successful_trains, failed_trains
    
    def generate_training_report(self, successful_trains, failed_trains):
        """Generate a training report"""
        report_path = os.path.join(MODEL_DIR, "training_report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Stock Price Prediction Model Training Report\n")
            f.write("=" * 50 + "\n")
            f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Sequence Length: {self.sequence_length}\n")
            f.write(f"Future Days: {self.future_days}\n\n")
            
            f.write("Successful Trainings:\n")
            f.write("-" * 20 + "\n")
            for script in successful_trains:
                f.write(f"✅ {script}\n")
            
            f.write("\nFailed Trainings:\n")
            f.write("-" * 15 + "\n")
            for script in failed_trains:
                f.write(f"❌ {script}\n")
        
        print(f"📄 Training report saved to: {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Train stock prediction models")
    parser.add_argument("--stocks", type=str, help="Comma-separated list of stock symbols")
    parser.add_argument("--all", action="store_true", help="Train models for all allowed scripts")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--sequence-length", type=int, default=60, help="Sequence length for LSTM")
    parser.add_argument("--future-days", type=int, default=30, help="Number of future days to predict")
    
    args = parser.parse_args()
    
    # Determine which stocks to train
    if args.all:
        scripts_to_train = ALLOWED_SCRIPTS
    elif args.stocks:
        scripts_to_train = [s.strip().upper() for s in args.stocks.split(",")]
    else:
        print("❌ Please specify --stocks or --all")
        return
    
    # Validate scripts
    invalid_scripts = [s for s in scripts_to_train if s not in ALLOWED_SCRIPTS]
    if invalid_scripts:
        print(f"❌ Invalid scripts: {invalid_scripts}")
        print(f"✅ Allowed scripts: {ALLOWED_SCRIPTS}")
        return
    
    # Initialize trainer
    trainer = ModelTrainer(
        sequence_length=args.sequence_length,
        future_days=args.future_days
    )
    
    # Train models
    successful, failed = trainer.train_multiple_models(scripts_to_train, epochs=args.epochs)
    
    # Generate report
    trainer.generate_training_report(successful, failed)
    
    print(f"\n🎉 Training completed! Check {MODEL_DIR} for saved models.")

if __name__ == "__main__":
    main() 