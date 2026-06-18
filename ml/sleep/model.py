"""
Seren — Sleep Stage Classification: Model
===========================================
TFLite-friendly TCN + BiGRU seq2seq, identical to the Kaggle notebook
(seren_sleep_kaggle.ipynb) v3.2 SleepFeatureModel.

Input : [B, S, F]  (batch, SEQ_LEN epochs, NUM_FEATURES per epoch)
Output: [B, S, C]  (per-epoch logits; argmax last dim -> stage)

BatchNorm1d expects [B, C, L], so the forward transposes around the input norm,
the TCN, and the head norm. LSTM_HIDDEN / LSTM_LAYERS names are kept for
back-compat with prior runs — they parameterize the GRU.
"""

import torch
import torch.nn as nn

import config as cfg


class SleepFeatureModel(nn.Module):
    def __init__(self, n_features=None, n_classes=None):
        super().__init__()
        n_features = cfg.NUM_FEATURES if n_features is None else n_features
        n_classes = cfg.NUM_CLASSES if n_classes is None else n_classes

        self.in_norm = nn.BatchNorm1d(n_features)
        pad = cfg.TCN_KERNEL // 2
        self.tcn = nn.Sequential(
            nn.Conv1d(n_features, cfg.TCN_CHANNELS, cfg.TCN_KERNEL, padding=pad),
            nn.BatchNorm1d(cfg.TCN_CHANNELS), nn.GELU(), nn.Dropout(cfg.DROPOUT),
            nn.Conv1d(cfg.TCN_CHANNELS, cfg.TCN_CHANNELS, cfg.TCN_KERNEL, padding=pad),
            nn.BatchNorm1d(cfg.TCN_CHANNELS), nn.GELU(), nn.Dropout(cfg.DROPOUT),
        )
        self.rnn = nn.GRU(
            cfg.TCN_CHANNELS, cfg.LSTM_HIDDEN, num_layers=cfg.LSTM_LAYERS,
            batch_first=True, bidirectional=True,
            dropout=cfg.DROPOUT if cfg.LSTM_LAYERS > 1 else 0.0,
        )
        self.head_norm = nn.BatchNorm1d(cfg.LSTM_HIDDEN * 2)
        self.head = nn.Sequential(
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(cfg.LSTM_HIDDEN * 2, n_classes),
        )

    def forward(self, x):                       # x: [B, S, F]
        x = self.in_norm(x.transpose(1, 2)).transpose(1, 2)
        h = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        seq, _ = self.rnn(h)
        seq = self.head_norm(seq.transpose(1, 2)).transpose(1, 2)
        return self.head(seq)                   # [B, S, n_classes]

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(device=torch.device("cpu")):
    """Create and initialize the TCN+BiGRU model."""
    model = SleepFeatureModel().to(device)
    print(f"Model created: {model.count_parameters():,} trainable parameters | Device: {device}")
    return model
