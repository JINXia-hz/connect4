# Connect Four (Puissance 4) — Prolog + ML

[![CI](https://github.com/JINXia-hz/connect4/actions/workflows/ci.yml/badge.svg)](https://github.com/JINXia-hz/connect4/actions/workflows/ci.yml)

A Connect Four game implemented in SWI-Prolog, featuring a range of AI opponents
from pure random play to alpha-beta search with transposition tables and MCTS —
plus a machine-learning AI whose evaluation function is a neural network
trained by self-play (see "Machine learning" below).

## Credits

This repository is a **personal secondary development** based on a team course
project (H4104) originally built by:

> Esteban Wybouw · Lukas Vauvert · Eliott Frechard · Yanis Gallard ·
> Billal Belkho · Xia Jin · Louis Fiacre

The original project delivered the game engine and the classical AIs.
This fork focuses on:

- **Codebase standardization** — shared predicates factored into
  `src/core/board.pl`, duplicate/conflicting definitions removed, naming
  unified to English `snake_case`, comments rewritten in English.
- **Testing & CI** — the ad-hoc test script was migrated to the standard
  [plunit](https://www.swi-prolog.org/pldoc/doc_for?object=section(%27packages/plunit.html%27))
  framework with expanded coverage, running green on GitHub Actions.
- **Machine learning** — a neural-network evaluation function trained by
  self-play, plugged into the existing alpha-beta search.

## Requirements

- [SWI-Prolog](https://www.swi-prolog.org/) 9+ (developed on 10.0), with the
  Janus Python interface (bundled with the standard installers)
- *Only for the NN AI*: Python 3.10+ visible to Janus, plus
  `pip install -r requirements.txt` (numpy + CPU torch). A trained
  `ml/model.pt` must exist (`py ml/train.py`); without it the NN AI cannot
  play, but all other AIs and the test suite work fine.

## How to play

```sh
swipl main.pl
```

then type:

```prolog
?- jouer.
```

The menu offers Human vs Human, Human vs AI, and AI vs AI.
When playing, enter the column number followed by a dot (e.g. `4.`).

## The AIs

| Menu name | Atom (in code) | Description |
|---|---|---|
| Random | `computer_random` | Plays uniformly at random. |
| Random+ | `computer_evolue` | Random, but takes any immediate win. |
| Random++ | `computer_random_plus` | Random, but takes immediate wins and blocks immediate losses. |
| Alpha-Beta (pedagogical) | `computer_alphabeta` | Negamax alpha-beta at depth 8 (falls back to 7, then 5 on stack overflow) with a naive win/loss-only evaluation. |
| Minimax (strong) | `computer` | Alpha-beta with transposition table, adaptive depth, quiescence-style extension, and a window-based heuristic (2/3-in-a-row, playability under gravity, center control). |
| MCTS | `computer_mcts` | Monte-Carlo Tree Search (UCB1, 3000 iterations/move by default; settable with `set_mcts_iterations/1`). |
| Neural Network | `computer_nn` | Alpha-beta (depth 3) with a neural-network leaf evaluation via Janus. |

### A note on the strong minimax cache

`src/ai/minimax.pl` keeps a transposition table across moves within a game —
this is intended. Clearing it every move would make the AI amnesic (slower and
weaker). The game loop already cleans it between games. If you script *repeated
independent runs* from a single session, uncomment the `clean_cache` call at
the top of `ia_play/4`.

## Running the tests

```sh
swipl -g "run_tests,halt(0)" -t "halt(1)" tests/tests.pl
```

All 57 tests run non-interactively; the exit code is non-zero on failure
(this is exactly what CI runs). From the interactive prompt, `tests.` works too.

Coverage includes: victory detection (vertical/horizontal/both diagonals),
draw detection, board utilities and gravity, `check_winning_move/4`, and
behavioral tests for every AI (immediate wins, immediate blocks, valid columns),
plus benchmark and self-play smoke tests.

## Benchmarking

```sh
swipl src/tools/benchmark.pl
```

```prolog
?- benchmark(computer, computer_mcts, 10, WinsA, WinsB, Draws).
```

`benchmark/6` plays N games, randomly alternating who moves first, and returns
the win/draw tallies.

## Project layout

```
main.pl                  Entry point: loads the game (swipl main.pl, then jouer.)
src/
  core/board.pl          Shared predicates (board utils, gravity move, winning-move check)
  core/victory.pl        Win/draw detection
  game/puissance4.pl     Game loop, menus, display, AI dispatch
  ai/random.pl           Random / Random+ / Random++ AIs
  ai/alphabeta.pl        Pedagogical alpha-beta
  ai/minimax.pl          Strong minimax (TT, adaptive depth, quiescence)
  ai/mcts.pl             Monte-Carlo Tree Search (settable iteration count)
  ai/nn.pl               Neural-network AI (NN eval + alpha-beta, via Janus)
  selfplay/selfplay.pl   Self-play data generation (heuristic / nn / mcts agents)
  tools/benchmark.pl     Automated AI-vs-AI benchmarking
tests/tests.pl           plunit test suite
ml/                      Python package (encode / model / bridge / train / gen_data)
data/                    Generated self-play datasets (git-ignored)
```

All Prolog files are consult-based into the shared `user` namespace (no module
system). Relative `consult` paths resolve against the including file's
directory, so loading works from any current working directory; the only
CWD-relative paths are the generated data files (`data/...`).

## Machine learning: the NN AI

An AlphaZero-lite pipeline: a dual-head CNN (PyTorch) predicts, from the
current player's perspective, both the game outcome (**value head**, used as
the leaf evaluation of a dedicated alpha-beta search in `src/ai/nn.pl`) and a
probability distribution over the 7 columns (**policy head**, used to order
moves inside the search — better ordering means more alpha-beta cutoffs, so
the same time budget searches deeper). Prolog calls Python in-process via
SWI-Prolog's Janus bridge (`py_call(bridge:eval(Board, Player), Value)`,
`py_call(bridge:policy(Board, Player), Probs)`).

The training loop:

```sh
# 1. Generate games with the MCTS teacher, with random openings and
#    epsilon-exploration. Every position is recorded with the final result
#    (value label) and the teacher's chosen move (policy label). The NN never
#    learns from the hand-written heuristic.
py ml/gen_data.py --games 300 --out data/games_v1.txt \
    --player1 mcts:400 --player2 mcts:400 --workers 5

# 2. Train both heads on all data/*.txt (loss = value MSE + policy
#    cross-entropy).
py ml/train.py

# 3. Iterate: the NN AI now plays itself, generating stronger data; retrain
#    on the accumulated dataset.
py ml/gen_data.py --games 200 --out data/games_v2.txt \
    --player1 nn:3 --player2 nn:3 --workers 5
py ml/train.py
```

Player specs for `gen_data.py` are `<kind>:<param>`: `mcts:N` (N iterations
per move), `heuristic:D` (alpha-beta depth D), `nn:D` (NN search depth D).
`--workers N` splits the games across N parallel SWI-Prolog processes
(near-linear speedup; each worker is single-threaded).

The trained model (`ml/model.pt`, git-ignored) is loaded lazily by
`ml/bridge.py`; evaluations and policy queries are cached across the search.
If Python/torch is unavailable, the NN AI falls back to the hand-written
heuristic and center-based move ordering instead of crashing. In the game
menu the NN AI is option 7 (`computer_nn`).

### Measured results (this machine, benchmark/6 with random first player)

v1/v2 were trained on heuristic-minimax self-play labels; **v3+ are trained
purely on MCTS-teacher data**, so the network learns from simulation
statistics rather than from the hand-written heuristic. v5 adds the policy
head and policy-guided move ordering.

| Matchup | v1–v2 (heuristic data) | v3 (MCTS 200 g) | v4 (MCTS ~700 g) | v5 (+policy head) |
|---|---|---|---|---|
| NN vs Random++ (20 games) | 17–18 wins | 17–2–1 | 18–1–1 | **19–1–0** |
| NN vs strong heuristic Minimax | 0–6–0 (6 g) | 0–6–0 (6 g) | 0–6–0 (6 g) | **15–5–0 (20 g)** |
| NN vs heuristic, **equal depth 3** (40 games, both colors) | 0–39–1 | 5–34–1 | 7–31–2 | 9–27–4 |

The milestone: **v5 is the first version to beat the original hand-tuned
minimax overall** — 15 wins to 5 over 20 games, despite the minimax searching
at adaptive depth 5–7 with quiescence extension. The policy head was the key:
ordering moves by the learned policy prunes the alpha-beta tree so
effectively that the NN search at nominal depth 4 outperforms the deeper
heuristic search, and the measured win rate at equal depth (9 wins / 4 draws
out of 40) confirms the learned evaluation itself keeps improving with more
MCTS data. The pipeline (self-play → train → plug into search → benchmark →
iterate) works end-to-end and each piece is measurable.

### Ideas for going further

- More self-play iterations (the loop above is designed for it); raise the
  MCTS teacher's iteration count (`mcts:1600`+).
- NN-vs-NN self-play iterations (AlphaZero's actual loop; so far the teacher
  has been pure MCTS).
- Use the policy head as MCTS priors (PUCT) instead of only for move ordering.
- Tune `nn_depth` in `src/ai/nn.pl` (speed/strength trade-off).
