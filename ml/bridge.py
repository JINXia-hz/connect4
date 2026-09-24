"""Janus bridge: called from Prolog via py_call(bridge:eval(Board, Player), V).

Janus converts the Prolog board (list of 7 columns of 6 atoms) into nested
Python lists of strings, and atoms 'x'/'o' into str. The model is lazy-loaded
once per Prolog process; evaluations are cached because alpha-beta revisits
positions through the transposition table miss path.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from encode import encode_prolog_board  # noqa: E402
from model import Connect4ValueNet  # noqa: E402

_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.pt')
_model = None
_cache = {}


def _load_model():
    global _model
    if _model is None:
        if not os.path.exists(_MODEL_PATH):
            raise FileNotFoundError(
                f"no trained model at {_MODEL_PATH} — run ml/train.py first")
        _model = Connect4ValueNet()
        _model.load_state_dict(torch.load(_MODEL_PATH, map_location='cpu',
                                          weights_only=True))
        _model.eval()
    return _model


def _to_key(board, player):
    return tuple(tuple(str(cell) for cell in col) for col in board), str(player)


def eval(board, player):
    """Prolog board + side-to-move mark -> float value in [-1, 1]."""
    key = _to_key(board, player)
    if key in _cache:
        return _cache[key]
    x = torch.from_numpy(encode_prolog_board(board, str(player))).unsqueeze(0)
    with torch.no_grad():
        v = float(_load_model()(x).item())
    _cache[key] = v
    return v


def eval_batch(boards, player):
    """List of boards (same side to move) -> list of floats."""
    boards = [list(b) for b in boards]
    keys = [_to_key(b, player) for b in boards]
    missing = [i for i, k in enumerate(keys) if k not in _cache]
    if missing:
        xs = np.stack([encode_prolog_board(boards[i], str(player)) for i in missing])
        with torch.no_grad():
            vs = _load_model()(torch.from_numpy(xs)).tolist()
        for i, v in zip(missing, vs):
            _cache[keys[i]] = float(v)
    return [_cache[k] for k in keys]


def reset_cache():
    _cache.clear()
    return True
