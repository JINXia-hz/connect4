"""Train the dual-head network on self-play data.

Usage: py ml/train.py [--data data] [--out ml/model.pt] [--epochs 20]

Reads every *.txt in the data directory. Line format (see ml/encode.py):
"<board42> <player> <result> [<move>]" — result in {1, -1, 0} from the
perspective of the player to move; move is the column (1-7) chosen by the
teacher at that position, or '-' when the move was random (opening or
epsilon exploration). 3-field legacy lines are accepted (value-only).
Loss = MSE(value) + cross_entropy(policy_logits, move) over the
move-labeled samples, with equal weights.
"""

import argparse
import glob
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from encode import encode_board_str
from model import Connect4Net


def load_dataset(data_dir):
    """Returns (xs, ys, ms): encoded boards, value labels, move labels.
    ms holds 0-based target columns, or -1 for value-only samples."""
    records = {}
    for path in sorted(glob.glob(os.path.join(data_dir, '*.txt'))):
        with open(path, encoding='utf-8') as f:
            for line in f:
                parts = line.split()
                if len(parts) == 3:
                    board, player, result = parts
                    move = None
                elif len(parts) == 4:
                    board, player, result, move_s = parts
                    move = None if move_s == '-' else int(move_s) - 1
                else:
                    continue
                records[(board, player)] = (float(result), move)  # dedupe, latest wins
    if not records:
        raise SystemExit(f"no training data found in {data_dir!r}")
    xs = np.stack([encode_board_str(b, p) for b, p in records])
    ys = np.array([r for r, _ in records.values()], dtype=np.float32)
    ms = np.array([-1 if m is None else m for _, m in records.values()],
                  dtype=np.int64)
    return torch.from_numpy(xs), torch.from_numpy(ys), torch.from_numpy(ms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data')
    ap.add_argument('--out', default=os.path.join('ml', 'model.pt'))
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    xs, ys, ms = load_dataset(args.data)
    n = len(ys)
    n_labeled = int((ms >= 0).sum())
    perm = torch.randperm(n)
    split = max(1, int(n * 0.9))
    tr, va = perm[:split], perm[split:]

    model = Connect4Net()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    mse_fn = nn.MSELoss()

    print(f"dataset: {n} positions ({len(tr)} train / {len(va)} val), "
          f"{n_labeled} with move labels")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, batches = 0.0, 0
        shuffled = tr[torch.randperm(len(tr))]
        for i in range(0, len(shuffled), args.batch):
            idx = shuffled[i:i + args.batch]
            opt.zero_grad()
            values, logits = model(xs[idx])
            loss = mse_fn(values, ys[idx])
            labeled = ms[idx] >= 0
            if labeled.any():
                loss = loss + F.cross_entropy(logits[labeled], ms[idx][labeled])
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            batches += 1
        model.eval()
        with torch.no_grad():
            values, logits = model(xs[va])
            val_loss = mse_fn(values, ys[va]).item()
            va_labeled = ms[va] >= 0
            if va_labeled.any():
                pred = logits[va_labeled].argmax(dim=-1)
                acc = (pred == ms[va][va_labeled]).float().mean().item() * 100
                pol_msg = (f"policy_top1={acc:.1f}% "
                           f"({int(va_labeled.sum())} labeled val samples)")
            else:
                pol_msg = "policy: no labeled samples"
        print(f"epoch {epoch:3d}  train_loss={epoch_loss / batches:.4f}  "
              f"val_mse={val_loss:.4f}  {pol_msg}")

    torch.save(model.state_dict(), args.out)
    print(f"saved -> {args.out}")


if __name__ == '__main__':
    main()
