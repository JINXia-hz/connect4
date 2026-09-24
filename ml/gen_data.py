"""Self-play data generation driver.

Spawns SWI-Prolog running src/selfplay/selfplay.pl to generate games, then
(optionally) retrains. One iteration of the training loop looks like:

    py ml/gen_data.py --games 300 --out data/games_v1.txt \
        --player1 mcts:800 --player2 mcts:800              # MCTS teacher
    py ml/train.py

    py ml/gen_data.py --games 300 --out data/games_v2.txt \
        --player1 nn:3 --player2 nn:3                      # later iterations
    py ml/train.py

Player spec: "<kind>:<param>" where kind is 'mcts' (Monte-Carlo teacher,
param = iterations per move, e.g. mcts:800), 'heuristic' (hand-written
eval, param = search depth) or 'nn' (neural eval via Janus; requires a
trained ml/model.pt).
"""

import argparse
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--games', type=int, default=300)
    ap.add_argument('--out', required=True, help="output file, e.g. data/games_v1.txt")
    ap.add_argument('--player1', default='heuristic:3')
    ap.add_argument('--player2', default='heuristic:3')
    ap.add_argument('--epsilon', type=float, default=0.10,
                    help="probability of a random move (exploration)")
    ap.add_argument('--open-random', type=int, default=4,
                    help="fully random opening moves per game")
    args = ap.parse_args()

    goal = (
        "consult('src/selfplay/selfplay.pl'), "
        f"gen_games({args.games}, '{args.out}', "
        f"spec(({args.player1.replace(':', ',')}), "
        f"({args.player2.replace(':', ',')}), "
        f"{args.epsilon}, {args.open_random})), "
        "halt(0)"
    )
    cmd = ['swipl', '-g', goal, '-t', 'halt(1)']
    print('running:', ' '.join(cmd), flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        sys.exit(rc)
    print(f"done -> {args.out}")


if __name__ == '__main__':
    main()
