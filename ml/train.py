"""Train the value network on self-play data.

Usage: py ml/train.py [--data data] [--out ml/model.pt] [--epochs 20]

Reads every *.txt in the data directory. Line format (see ml/encode.py):
"<board42> <player> <result>" — result in {1, -1, 0} from the perspective
of the player to move.
"""

import argparse
import glob
import os
import random

import numpy as np
import torch
import torch.nn as nn

from encode import encode_board_str
from model import Connect4ValueNet


def load_dataset(data_dir):
    records = {}
    for path in sorted(glob.glob(os.path.join(data_dir, '*.txt'))):
        with open(path, encoding='utf-8') as f:
            for line in f:
                parts = line.split()
                if len(parts) != 3:
                    continue
                board, player, result = parts
                records[(board, player)] = float(result)  # dedupe, latest wins
    if not records:
        raise SystemExit(f"no training data found in {data_dir!r}")
    xs = np.stack([encode_board_str(b, p) for b, p in records])
    ys = np.array(list(records.values()), dtype=np.float32)
    return torch.from_numpy(xs), torch.from_numpy(ys)


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

    xs, ys = load_dataset(args.data)
    n = len(ys)
    perm = torch.randperm(n)
    split = max(1, int(n * 0.9))
    tr, va = perm[:split], perm[split:]

    model = Connect4ValueNet()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    print(f"dataset: {n} positions ({len(tr)} train / {len(va)} val)")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, batches = 0.0, 0
        shuffled = tr[torch.randperm(len(tr))]
        for i in range(0, len(shuffled), args.batch):
            idx = shuffled[i:i + args.batch]
            opt.zero_grad()
            loss = loss_fn(model(xs[idx]), ys[idx])
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            batches += 1
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(xs[va]), ys[va]).item()
        print(f"epoch {epoch:3d}  train_mse={epoch_loss / batches:.4f}  "
              f"val_mse={val_loss:.4f}")

    torch.save(model.state_dict(), args.out)
    print(f"saved -> {args.out}")


if __name__ == '__main__':
    main()
