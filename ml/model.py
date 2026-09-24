"""Value network for Connect Four.

Input: (batch, 2, 6, 7) — current player's pieces / opponent's pieces.
Output: (batch,) value in [-1, 1], the predicted game outcome from the
current player's perspective (trained on final results: win +1, loss -1,
draw 0).
"""

import torch
import torch.nn as nn

from encode import ROWS, COLS


class Connect4ValueNet(nn.Module):
    def __init__(self, channels=64, hidden=128):
        super().__init__()
        self.conv1 = nn.Conv2d(2, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(channels * ROWS * COLS, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        return torch.tanh(self.fc2(x)).squeeze(-1)
