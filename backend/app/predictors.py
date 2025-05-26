import torch
from .config import settings
from model.cnn_lstm_hybrid import ParallelCNNLSTM

# Cache model
_model = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    global _model
    if _model is None:
        # adjust input_dim=4 features
        m = ParallelCNNLSTM(input_dim=4).to(_device)
        m.load_state_dict(torch.load("model.pth", map_location=_device))
        m.eval()
        _model = m
    return _model


def predict_batch(model, windows, symbol=None):
    # windows: numpy array of shape [N, past, features]
    import numpy as np
    import torch
    x = torch.tensor(windows, dtype=torch.float32).to(_device)
    with torch.no_grad():
        preds = model(x).cpu().numpy()
    # last prediction is for most recent window
    return preds[-1]