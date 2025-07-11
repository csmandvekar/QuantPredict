# Stock Price Prediction Training Pipeline

This directory contains scripts to train and manage stock price prediction models for your QuantPredict application.

## 📁 Structure

```
training_pipeline/
├── train_models.py          # Main training script
├── add_new_stock.py         # Add new stocks to the system
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd training_pipeline
pip install -r requirements.txt
```

### 2. Train Models for Existing Stocks

```bash
# Train all configured stocks
python train_models.py --all --epochs 50

# Train specific stocks
python train_models.py --stocks INFY,TCS,RELIANCE --epochs 100

# Train with custom parameters
python train_models.py --stocks INFY --epochs 200 --sequence-length 90 --future-days 60
```

### 3. Add New Stocks

```bash
# Add a single stock
python add_new_stock.py --stock AAPL --epochs 50

# Add multiple stocks
python add_new_stock.py --stock MSFT,GOOGL,AMZN --epochs 100
```

## 📊 How It Works

### Data Flow

1. **Data Fetching**: Uses `yfinance` to fetch historical stock data
2. **Preprocessing**: Creates technical indicators and normalizes data
3. **Sequence Creation**: Prepares LSTM sequences for training
4. **Model Training**: Trains CNN-LSTM model with validation
5. **Model Saving**: Saves trained models to `backend/pretrained_models/`

### Model Architecture

- **CNN-LSTM Hybrid**: Combines CNN for feature extraction and LSTM for sequence modeling
- **Technical Indicators**: Uses 17+ technical indicators (RSI, MACD, Bollinger Bands, etc.)
- **Sequence Length**: Default 60 days of historical data
- **Prediction Horizon**: Default 30 days into the future

### Technical Indicators Used

- Moving Averages (7, 21, 50 days)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Price change percentages
- Volume indicators
- High-Low percentages

## 🎯 Usage Examples

### Train All Current Stocks

```bash
python train_models.py --all --epochs 100
```

This will train models for all stocks in `backend/app/core/config.py` (currently INFY, TCS, RELIANCE).

### Train Specific Stocks

```bash
python train_models.py --stocks INFY,TCS --epochs 150
```

### Add New Stock to System

```bash
python add_new_stock.py --stock AAPL --epochs 75
```

This will:
1. Add AAPL to the allowed scripts in config
2. Fetch historical data for AAPL
3. Train a model for AAPL
4. Save the model to `backend/pretrained_models/`

### Custom Training Parameters

```bash
python train_models.py \
    --stocks INFY \
    --epochs 200 \
    --sequence-length 90 \
    --future-days 60
```

## 📈 Model Performance

The training script provides detailed metrics:

- **MSE (Mean Squared Error)**: Overall prediction accuracy
- **MAE (Mean Absolute Error)**: Average prediction error
- **Training Loss**: Model convergence during training
- **Validation Loss**: Model generalization performance

## 💾 Model Storage

Trained models are saved as:
- `{STOCK}_model.pkl` in `backend/pretrained_models/`
- Training reports saved as `training_report.txt`

## 🔧 Configuration

### Current Stocks (in `backend/app/core/config.py`)
```python
ALLOWED_SCRIPTS = ["INFY", "TCS", "RELIANCE"]
```

### Model Parameters
- **Sequence Length**: 60 days (configurable)
- **Future Days**: 30 days (configurable)
- **Validation Split**: 20% of data
- **Features**: 17 technical indicators

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're running from the `training_pipeline` directory
2. **Data Fetching Errors**: Check internet connection and stock symbol validity
3. **Memory Issues**: Reduce batch size or sequence length for large datasets
4. **GPU Issues**: Install `tensorflow-gpu` for GPU acceleration

### Error Messages

- `ModuleNotFoundError`: Install missing dependencies with `pip install -r requirements.txt`
- `No data found for {symbol}`: Check stock symbol and internet connection
- `CUDA out of memory`: Reduce batch size or use CPU-only TensorFlow

## 📝 Logs and Reports

The training pipeline generates:
- **Console output**: Real-time training progress
- **Training report**: Summary saved to `backend/pretrained_models/training_report.txt`
- **Model files**: Saved models ready for prediction

## 🔄 Integration with Web App

After training:
1. Models are automatically available in your web application
2. New stocks appear in the dropdown
3. Predictions use the trained models
4. No restart required for the web app

## 🎉 Success Indicators

- Models saved to `backend/pretrained_models/`
- Training report generated
- New stocks appear in web app dropdown
- Predictions work without errors

## 📞 Support

If you encounter issues:
1. Check the console output for error messages
2. Verify all dependencies are installed
3. Ensure stock symbols are valid
4. Check internet connection for data fetching 