"""League-style iterative training orchestrator.

Each round:
1. TRAINING GAMES - three pairings x games-per-pairing games (epsilon 0.1,
   open-random 4), split across workers, random sides per chunk:
   nn:nnDepth vs heuristic:mmDepth, nn:nnDepth vs mcts:iters,
   heuristic:mmDepth vs mcts:iters.  -> data/league_r{k}.txt
2. TRAIN - ml/train.py on ALL data/*.txt, checkpoint -> ml/model_r{k}.pt
3. EVALUATE - recording-free play_match/7 matches (10 games each):
   NN vs heuristic:mmDepth, NN vs mcts:iters, NN vs original (fixed ruler).
4. ESCALATE - if the NN win rate (wins + half draws) reaches
   --escalate-threshold, the opponent gets stronger (mmDepth + 1, cap 7;
   iters x 1.5, cap 1600).
5. STOP - NN win rate vs original >= --stop-threshold in two consecutive
   rounds -> CONVERGED.
6. The best round by original-ruler win rate is copied to ml/model_best.pt.
7. One machine-readable line per round is appended to league_log.txt.

Run from the project root:  py ml/league.py [--rounds 8] [--workers 5]
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gen_data import generate_games, merge_parts, spec_to_prolog  # noqa: E402

MODEL = os.path.join('ml', 'model.pt')
LOG_FILE = 'league_log.txt'
MATCH_RE = re.compile(r'MATCH_RESULT (\d+) (\d+) (\d+)')


def win_rate(wins, draws, games):
    """Win rate counting a draw as half a win."""
    return (wins + 0.5 * draws) / games if games else 0.0


def play_match(player1, player2, games, open_random):
    """Run a recording-free match; returns (wins1, wins2, draws)."""
    goal = (
        "consult('src/selfplay/selfplay.pl'), "
        f"play_match({spec_to_prolog(player1)}, {spec_to_prolog(player2)}, "
        f"{games}, {open_random}, W1, W2, D), halt(0)"
    )
    proc = subprocess.run(['swipl', '-g', goal, '-t', 'halt(1)'],
                          capture_output=True, text=True)
    m = MATCH_RE.search(proc.stdout)
    if proc.returncode != 0 or not m:
        sys.stderr.write(proc.stdout[-2000:])
        sys.stderr.write(proc.stderr[-2000:])
        raise SystemExit(f"play_match failed: {player1} vs {player2}")
    for line in proc.stdout.splitlines():
        if line.startswith('match game'):
            print(f"    {line}", flush=True)
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rounds', type=int, default=8)
    ap.add_argument('--games-per-pairing', type=int, default=8)
    ap.add_argument('--workers', type=int, default=5)
    ap.add_argument('--nn-depth', type=int, default=4)
    ap.add_argument('--minimax-depth', type=int, default=4)
    ap.add_argument('--mcts-iters', type=int, default=400)
    ap.add_argument('--escalate-threshold', type=float, default=0.6)
    ap.add_argument('--stop-threshold', type=float, default=0.7)
    ap.add_argument('--eval-games', type=int, default=10)
    ap.add_argument('--epochs', type=int, default=25)
    args = ap.parse_args()

    max_mm_depth, max_iters = 7, 1600
    mm_depth, iters = args.minimax_depth, args.mcts_iters
    nn_spec = f"nn:{args.nn_depth}"

    best_rate, best_round = -1.0, None
    consecutive_stop = 0
    history = []

    header = (f"{'rnd':>3} {'mmD':>3} {'iter':>4} | {'vs_heur':>7} "
              f"{'vs_mcts':>7} {'vs_orig':>7} | notes")
    print(header, flush=True)

    for k in range(1, args.rounds + 1):
        t_round = time.time()
        notes = []

        # --- 1. Training games ---
        t0 = time.time()
        out = os.path.join('data', f'league_r{k}.txt')
        pairings = [
            (nn_spec, f"heuristic:{mm_depth}"),
            (nn_spec, f"mcts:{iters}"),
            (f"heuristic:{mm_depth}", f"mcts:{iters}"),
        ]
        parts = []
        for tag, (p1, p2) in zip('abc', pairings):
            part = f"{out}.pair{tag}"
            print(f"[round {k}] games {p1} vs {p2} "
                  f"({args.games_per_pairing} games)", flush=True)
            generate_games(args.games_per_pairing, part, p1, p2,
                           0.1, 4, args.workers, random_sides=True)
            parts.append(part)
        merge_parts(parts, out)
        t_games = time.time() - t0

        # --- 2. Train + checkpoint ---
        t0 = time.time()
        train_cmd = [sys.executable, os.path.join('ml', 'train.py'),
                     '--epochs', str(args.epochs)]
        # Warm-start from the previous round's weights (true iteration);
        # round 1 starts from whatever ml/model.pt currently holds.
        if os.path.exists(os.path.join('ml', 'model.pt')):
            train_cmd += ['--init', os.path.join('ml', 'model.pt')]
        rc = subprocess.call(train_cmd)
        if rc != 0:
            raise SystemExit(f"training failed in round {k}")
        checkpoint = os.path.join('ml', f'model_r{k}.pt')
        shutil.copyfile(MODEL, checkpoint)
        t_train = time.time() - t0

        # --- 3. Evaluate ---
        t0 = time.time()
        print(f"[round {k}] eval: {nn_spec} vs heuristic:{mm_depth}", flush=True)
        w_h, _, d_h = play_match(nn_spec, f"heuristic:{mm_depth}",
                                 args.eval_games, 4)
        print(f"[round {k}] eval: {nn_spec} vs mcts:{iters}", flush=True)
        w_m, _, d_m = play_match(nn_spec, f"mcts:{iters}",
                                 args.eval_games, 4)
        print(f"[round {k}] eval: {nn_spec} vs original (ruler)", flush=True)
        w_o, _, d_o = play_match(nn_spec, "original:0",
                                 args.eval_games, 4)
        t_eval = time.time() - t0

        r_h = win_rate(w_h, d_h, args.eval_games)
        r_m = win_rate(w_m, d_m, args.eval_games)
        r_o = win_rate(w_o, d_o, args.eval_games)

        # --- 4. Escalate ---
        if r_h >= args.escalate_threshold and mm_depth < max_mm_depth:
            mm_depth += 1
            notes.append(f"escalate:mmDepth->{mm_depth}")
        if r_m >= args.escalate_threshold and iters < max_iters:
            iters = min(max_iters, int(iters * 1.5))
            notes.append(f"escalate:mctsIters->{iters}")

        # --- 5. Stop rule ---
        if r_o >= args.stop_threshold:
            consecutive_stop += 1
            notes.append(f"stop-streak:{consecutive_stop}/2")
        else:
            consecutive_stop = 0

        # --- 6. Best checkpoint tracking ---
        if r_o > best_rate:
            best_rate, best_round = r_o, k
            notes.append("new-best")

        # --- 7. Log + table row ---
        row = (f"{k:>3} {mm_depth:>3} {iters:>4} | {r_h:>7.2f} "
               f"{r_m:>7.2f} {r_o:>7.2f} | {' '.join(notes)}")
        print(row, flush=True)
        history.append(row)
        with open(LOG_FILE, 'a', encoding='utf-8') as log:
            log.write(f"round={k} nn_depth={args.nn_depth} mm_depth={mm_depth} "
                      f"mcts_iters={iters} wr_heuristic={r_h:.3f} "
                      f"wr_mcts={r_m:.3f} wr_original={r_o:.3f} "
                      f"games_s={t_games:.1f} train_s={t_train:.1f} "
                      f"eval_s={t_eval:.1f} round_s={time.time() - t_round:.1f} "
                      f"notes={' '.join(notes) or '-'}\n")

        if consecutive_stop >= 2:
            print(f"CONVERGED: NN win rate vs original >= "
                  f"{args.stop_threshold} for two consecutive rounds",
                  flush=True)
            break

    # --- Best checkpoint ---
    if best_round is not None:
        shutil.copyfile(os.path.join('ml', f'model_r{best_round}.pt'),
                        os.path.join('ml', 'model_best.pt'))
        print(f"best round: {best_round} (original-ruler win rate "
              f"{best_rate:.2f}) -> ml/model_best.pt", flush=True)


if __name__ == '__main__':
    main()
