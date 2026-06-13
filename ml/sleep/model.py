"""
Seren — Sleep Stage Classification: Model Architecture
========================================================
CNN-BiLSTM hybrid for 5-class sleep stage classification
from raw PPG + accelerometer signals.

Architecture:
  1D CNN feature extractor → BiLSTM sequence model → classifier head
  + auxiliary epoch-level features concatenated before classifier
"""

import torch
import torch.nn as nn

import config as cfg


class SleepStageCNN(nn.Module):
    """1D CNN feature extractor for raw wearable signals."""

    def __init__(self):
        super().__init__()

        layers = []
        in_channels = cfg.NUM_RAW_CHANNELS

        for out_ch, kernel, stride in zip(cfg.CNN_CHANNELS, cfg.CNN_KERNELS, cfg.CNN_STRIDES):
            layers.extend([
                nn.Conv1d(in_channels, out_ch, kernel_size=kernel, stride=stride, padding=kernel // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout(cfg.DROPOUT),
            ])
            in_channels = out_ch

        layers.append(nn.AdaptiveAvgPool1d(32))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, channels=2, samples=3000]
        return self.encoder(x)  # [batch, 128, 32]


class SleepStageModel(nn.Module):
    """
    Full sleep stage classification model.

    Input:
        raw: [batch, 2, 3000]  — BVP + ACC magnitude at 100Hz
        aux: [batch, 6]        — epoch-level stats (HR, ACC, TEMP)

    Output:
        logits: [batch, 5]     — Wake, N1, N2, N3, REM
    """

    def __init__(self):
        super().__init__()

        self.cnn = SleepStageCNN()

        self.lstm = nn.LSTM(
            input_size=cfg.CNN_CHANNELS[-1],
            hidden_size=cfg.LSTM_HIDDEN,
            num_layers=cfg.LSTM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.LSTM_DROPOUT if cfg.LSTM_LAYERS > 1 else 0.0,
        )

        lstm_out_size = cfg.LSTM_HIDDEN * 2  # bidirectional
        classifier_in = lstm_out_size + cfg.NUM_AUX_FEATURES

        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, cfg.CLASSIFIER_HIDDEN),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.DROPOUT + 0.1),
            nn.Linear(cfg.CLASSIFIER_HIDDEN, cfg.NUM_CLASSES),
        )

    def forward(self, raw: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        # CNN feature extraction
        cnn_out = self.cnn(raw)                    # [batch, 128, 32]
        cnn_out = cnn_out.permute(0, 2, 1)         # [batch, 32, 128]

        # BiLSTM sequence modeling
        lstm_out, _ = self.lstm(cnn_out)            # [batch, 32, 128]
        lstm_last = lstm_out[:, -1, :]              # [batch, 128]

        # Concatenate with auxiliary features
        combined = torch.cat([lstm_last, aux], dim=1)  # [batch, 134]

        # Classification
        logits = self.classifier(combined)          # [batch, 5]
        return logits

    def predict_proba(self, raw: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        """Get class probabilities (for inference)."""
        logits = self.forward(raw, aux)
        return torch.softmax(logits, dim=1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(device: torch.device = torch.device("cpu")) -> SleepStageModel:
    """Create and initialize the model."""
    model = SleepStageModel().to(device)
    print(f"Model created: {model.count_parameters():,} trainable parameters")
    return model
