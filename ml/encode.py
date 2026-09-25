"""Board encoding shared by training and inference.

Board wire format (contract with the Prolog side):
- A board is a 42-character string, column-major: columns 1..7, each
  column rows 1..6 from bottom (row 1) to top (row 6).
- Cells are 'x', 'o' or '_' (empty).
- A position record line is: "<board42> <player> <result> [<move>]" where player is
  the side to move ('x'/'o'), result is the final game outcome from that
  player's perspective: 1 (win), -1 (loss), 0 (draw), and move is the
  teacher-chosen column 1-7, or '-' when the move was random (policy target
  unavailable). 3-field legacy lines are value-only.

Tensor encoding: float32 array of shape (2, 6, 7) — [channel][row][col],
row 0 = bottom. Channel 0 = current player's pieces, channel 1 = opponent's.
Canonicalizing to the current player's perspective lets the network learn a
single value function compatible with negamax-style sign flipping.
"""

import numpy as np

ROWS = 6
COLS = 7
EMPTY = '_'


def board_to_grid(board_str):
    """42-char column-major string -> 2D list grid[row][col], row 0 = bottom."""
    if len(board_str) != ROWS * COLS:
        raise ValueError(f"expected {ROWS * COLS} chars, got {len(board_str)}")
    grid = [[EMPTY] * COLS for _ in range(ROWS)]
    for c in range(COLS):
        for r in range(ROWS):
            grid[r][c] = board_str[c * ROWS + r]
    return grid


def grid_to_board_str(grid):
    """Inverse of board_to_grid."""
    return ''.join(grid[r][c] for c in range(COLS) for r in range(ROWS))


def prolog_board_to_str(board):
    """Prolog board (list of 7 columns, each a list of 6 cell strings,
    bottom-up) -> 42-char wire string. Janus delivers atoms as str."""
    if len(board) != COLS:
        raise ValueError(f"expected {COLS} columns, got {len(board)}")
    return ''.join(str(cell) for col in board for cell in col)


def encode_grid(grid, player):
    """grid + side to move ('x'/'o') -> float32 (2, ROWS, COLS) tensor."""
    opp = 'o' if player == 'x' else 'x'
    mine = np.zeros((ROWS, COLS), dtype=np.float32)
    theirs = np.zeros((ROWS, COLS), dtype=np.float32)
    for r in range(ROWS):
        for c in range(COLS):
            if grid[r][c] == player:
                mine[r][c] = 1.0
            elif grid[r][c] == opp:
                theirs[r][c] = 1.0
    return np.stack([mine, theirs])


def encode_board_str(board_str, player):
    return encode_grid(board_to_grid(board_str), player)


def encode_prolog_board(board, player):
    return encode_board_str(prolog_board_to_str(board), player)
