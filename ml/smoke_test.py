"""Smoke tests for the ml package — run with: python ml/smoke_test.py
(Plain asserts, no pytest dependency; executed by CI.)
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from encode import (ROWS, COLS, board_to_grid, encode_board_str,  # noqa: E402
                    grid_to_board_str, prolog_board_to_str)
from model import Connect4ValueNet  # noqa: E402

EMPTY42 = '_' * (ROWS * COLS)

# Board string roundtrip.
assert grid_to_board_str(board_to_grid(EMPTY42)) == EMPTY42

# Column-major layout: 'x' at column 1 row 1 (bottom-left).
s = 'x' + '_' * (ROWS * COLS - 1)
grid = board_to_grid(s)
assert grid[0][0] == 'x' and grid[0][1] == '_' and grid[1][0] == '_'

# Encoding: current-player plane is canonical (x-to-move vs o-to-move swap planes).
t_x = encode_board_str(s, 'x')
t_o = encode_board_str(s, 'o')
assert t_x.shape == (2, ROWS, COLS)
assert t_x[0].sum() == 1 and t_x[1].sum() == 0      # 'x' is mine
assert t_o[1].sum() == 1 and t_o[0].sum() == 0      # 'x' is theirs
assert np.array_equal(t_x[0], t_o[1]) and np.array_equal(t_x[1], t_o[0])

# Prolog board (nested lists, as Janus delivers them) -> same wire string.
prolog_board = [['x', '_', '_', '_', '_', '_']] + [['_'] * ROWS for _ in range(COLS - 1)]
assert prolog_board_to_str(prolog_board) == s

# Model forward pass: shape and tanh range.
net = Connect4ValueNet()
out = net(torch.from_numpy(np.stack([t_x, t_o])))
assert out.shape == (2,)
assert bool((out.abs() <= 1.0).all())

print('smoke_test: all assertions passed')
