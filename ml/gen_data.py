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

Parallelism: games are independent, so `--workers N` splits the run into N
SWI-Prolog processes writing separate part files, then merges them into
--out. A good default is (physical cores - 1); each worker is
single-threaded.
"""

import argparse
import os
import subprocess
import sys


def run_chunk(games, out, player1, player2, epsilon, open_random):
    goal = (
        "consult('src/selfplay/selfplay.pl'), "
        f"gen_games({games}, '{out}', "
        f"spec(({player1.replace(':', ',')}), "
        f"({player2.replace(':', ',')}), "
        f"{epsilon}, {open_random})), "
        "halt(0)"
    )
    cmd = ['swipl', '-g', goal, '-t', 'halt(1)']
    return subprocess.call(cmd)


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
    ap.add_argument('--workers', type=int, default=1,
                    help="parallel SWI-Prolog processes (games are split "
                         "evenly; try physical cores minus one)")
    args = ap.parse_args()

    if args.workers <= 1:
        rc = run_chunk(args.games, args.out, args.player1, args.player2,
                       args.epsilon, args.open_random)
        if rc != 0:
            sys.exit(rc)
        print(f"done -> {args.out}")
        return

    # Parallel: split games into per-worker chunks, merge at the end.
    base, extra = divmod(args.games, args.workers)
    chunks = [base + (1 if i < extra else 0) for i in range(args.workers)]
    parts = [f"{args.out}.part{i}" for i in range(args.workers)]

    print(f"spawning {args.workers} workers: {chunks} games each", flush=True)
    procs = []
    for games, part in zip(chunks, parts):
        if games == 0:
            continue
        log = open(part + '.log', 'w')
        p = subprocess.Popen(
            ['swipl', '-g',
             "consult('src/selfplay/selfplay.pl'), "
             f"gen_games({games}, '{part}', "
             f"spec(({args.player1.replace(':', ',')}), "
             f"({args.player2.replace(':', ',')}), "
             f"{args.epsilon}, {args.open_random})), halt(0)",
             '-t', 'halt(1)'],
            stdout=log, stderr=subprocess.STDOUT)
        procs.append((p, log, part))

    failed = []
    for p, log, part in procs:
        rc = p.wait()
        log.close()
        print(f"worker {part}: exit {rc}", flush=True)
        if rc != 0:
            failed.append(part)
    if failed:
        print(f"FAILED workers: {failed} (see {failed[0]}.log)", file=sys.stderr)
        sys.exit(1)

    with open(args.out, 'w', encoding='utf-8', newline='\n') as out_f:
        for part in parts:
            if not os.path.exists(part):
                continue
            with open(part, encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        out_f.write(line if line.endswith('\n') else line + '\n')
            os.remove(part)
            os.remove(part + '.log')
    print(f"done -> {args.out}")


if __name__ == '__main__':
    main()
