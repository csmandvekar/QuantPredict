#!/usr/bin/env python3
"""
Add New Stock to Prediction System
==================================

This script helps you add new stocks to your prediction system:
1. Updates the ALLOWED_SCRIPTS configuration
2. Trains a model for the new stock
3. Saves the model to pretrained_models directory

Usage:
    python add_new_stock.py --stock AAPL --epochs 50
    python add_new_stock.py --stock MSFT,GOOGL --epochs 100
"""

import os
import sys
import argparse
import re
from datetime import datetime

# Add backend to path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from train_models import ModelTrainer

def validate_stock_symbol(symbol):
    """Validate stock symbol format"""
    # Basic validation - alphanumeric and common symbols
    if not re.match(r'^[A-Z]{1,5}$', symbol):
        return False
    return True

def update_config_file(new_stocks):
    """Update the config.py file to add new stocks"""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'backend', 'app', 'core', 'config.py')
    
    print(f"📝 Updating config file: {config_path}")
    
    # Read current config
    with open(config_path, 'r') as f:
        content = f.read()
    
    # Find ALLOWED_SCRIPTS line
    pattern = r'ALLOWED_SCRIPTS\s*=\s*\[(.*?)\]'
    match = re.search(pattern, content, re.DOTALL)
    
    if not match:
        print("❌ Could not find ALLOWED_SCRIPTS in config file")
        return False
    
    current_scripts = match.group(1)
    # Parse current scripts
    current_list = [s.strip().strip('"\'') for s in current_scripts.split(',') if s.strip()]
    
    # Add new stocks (avoid duplicates)
    updated_list = current_list.copy()
    for stock in new_stocks:
        if stock not in updated_list:
            updated_list.append(stock)
    
    # Create new ALLOWED_SCRIPTS line
    new_scripts_str = ', '.join([f'"{s}"' for s in updated_list])
    new_content = re.sub(pattern, f'ALLOWED_SCRIPTS = [{new_scripts_str}]', content, flags=re.DOTALL)
    
    # Write updated config
    with open(config_path, 'w') as f:
        f.write(new_content)
    
    print(f"✅ Added {len(new_stocks)} new stocks to config")
    print(f"📋 Current stocks: {', '.join(updated_list)}")
    
    return True

def add_new_stocks(stocks, epochs=50):
    """Add new stocks to the system"""
    print(f"🚀 Adding {len(stocks)} new stocks to the system...")
    
    # Validate stock symbols
    invalid_stocks = [s for s in stocks if not validate_stock_symbol(s)]
    if invalid_stocks:
        print(f"❌ Invalid stock symbols: {invalid_stocks}")
        print("   Stock symbols should be 1-5 uppercase letters")
        return False
    
    # Update config file
    if not update_config_file(stocks):
        return False
    
    # Train models for new stocks
    print(f"\n🎯 Training models for new stocks...")
    trainer = ModelTrainer()
    successful, failed = trainer.train_multiple_models(stocks, epochs=epochs)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"🎉 ADDITION SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successfully added: {len(successful)}")
    print(f"❌ Failed to add: {len(failed)}")
    
    if successful:
        print(f"✅ New stocks ready: {', '.join(successful)}")
    
    if failed:
        print(f"❌ Failed stocks: {', '.join(failed)}")
    
    return len(successful) > 0

def main():
    parser = argparse.ArgumentParser(description="Add new stocks to the prediction system")
    parser.add_argument("--stock", type=str, required=True, 
                       help="Stock symbol(s) to add (comma-separated for multiple)")
    parser.add_argument("--epochs", type=int, default=50, 
                       help="Number of training epochs")
    
    args = parser.parse_args()
    
    # Parse stock symbols
    stocks = [s.strip().upper() for s in args.stock.split(",")]
    
    print(f"📈 Adding stocks: {', '.join(stocks)}")
    print(f"🎯 Training epochs: {args.epochs}")
    
    # Add stocks
    success = add_new_stocks(stocks, epochs=args.epochs)
    
    if success:
        print(f"\n🎉 Successfully added new stocks to the system!")
        print(f"💡 You can now use these stocks in your web application.")
    else:
        print(f"\n❌ Failed to add stocks. Check the errors above.")

if __name__ == "__main__":
    main() 