"""Dual-head network for Connect Four.

Input: (batch, 2, 6, 7) — current player's pieces / opponent's pieces.
Outputs of forward(x):
- value: (batch,) in [-1, 1], the predicted game outcome from the
  current player's perspective (trained on final results: win +1,
  loss -1, draw 0).
- policy_logits: (batch, 7) raw logits over columns (softmax is applied
  at loss/inference time, not in forward). The policy target is the move
  chosen by the self-play teacher (AlphaZero-style).
"""

import torch
import torch.nn as nn

from encode import ROWS, COLS


class Connect4Net(nn.Module):
    def __init__(self, channels=64, hidden=128):
        super().__init__()
        # Shared trunk
        self.conv1 = nn.Conv2d(2, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(channels * ROWS * COLS, hidden)
        # Value head
        self.fc_value = nn.Linear(hidden, 1)
        # Policy head
        self.fc_policy = nn.Linear(hidden, COLS)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        value = torch.tanh(self.fc_value(x)).squeeze(-1)
        policy_logits = self.fc_policy(x)
        return value, policy_logits


# Backwards-compatible alias for the old name
Connect4ValueNet = Connect4Net
