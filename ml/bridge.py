"""Janus bridge: called from Prolog via py_call.

- bridge:eval(Board, Player)   -> float in [-1, 1]  (value head)
- bridge:policy(Board, Player) -> list of 7 floats  (policy head, softmax
  probabilities for ALL columns, including full ones — masking of full
  columns happens Prolog-side)
- bridge:reset_cache()         -> clears both caches

Janus converts the Prolog board (list of 7 columns of 6 atoms) into nested
Python lists of strings, and atoms 'x'/'o' into str. Python lists come back
to Prolog as lists (verified), so the 7-float policy list needs no packing.
The model is lazy-loaded once per Prolog process; evaluations and policies
are cached because alpha-beta revisits positions through the transposition
table miss path.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from encode import encode_prolog_board  # noqa: E402
from model import Connect4Net  # noqa: E402

_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.pt')
_model = None
_eval_cache = {}
_policy_cache = {}


def _load_model():
    global _model
    if _model is None:
        if not os.path.exists(_MODEL_PATH):
            raise FileNotFoundError(
                f"no trained model at {_MODEL_PATH} — run ml/train.py first")
        _model = Connect4Net()
        _model.load_state_dict(torch.load(_MODEL_PATH, map_location='cpu',
                                          weights_only=True))
        _model.eval()
    return _model


def _to_key(board, player):
    return tuple(tuple(str(cell) for cell in col) for col in board), str(player)


def _forward(board, player):
    """Encode + one forward pass -> (value float, policy logits tensor)."""
    x = torch.from_numpy(encode_prolog_board(board, str(player))).unsqueeze(0)
    with torch.no_grad():
        value, policy_logits = _load_model()(x)
    return float(value[0].item()), policy_logits[0]


def eval(board, player):
    """Prolog board + side-to-move mark -> float value in [-1, 1]."""
    key = _to_key(board, player)
    if key not in _eval_cache:
        value, _ = _forward(board, player)
        _eval_cache[key] = value
    return _eval_cache[key]


def policy(board, player):
    """Prolog board + side-to-move mark -> list of 7 softmax probabilities."""
    key = _to_key(board, player)
    if key not in _policy_cache:
        _, logits = _forward(board, player)
        _policy_cache[key] = torch.softmax(logits, dim=-1).tolist()
    return _policy_cache[key]


def eval_batch(boards, player):
    """List of boards (same side to move) -> list of floats."""
    boards = [list(b) for b in boards]
    keys = [_to_key(b, player) for b in boards]
    missing = [i for i, k in enumerate(keys) if k not in _eval_cache]
    if missing:
        xs = np.stack([encode_prolog_board(boards[i], str(player)) for i in missing])
        with torch.no_grad():
            vs, _ = _load_model()(torch.from_numpy(xs))
        for i, v in zip(missing, vs.tolist()):
            _eval_cache[keys[i]] = float(v)
    return [_eval_cache[k] for k in keys]


def reset_cache():
    _eval_cache.clear()
    _policy_cache.clear()
    return True
