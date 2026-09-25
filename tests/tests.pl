%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  tests/tests.pl                     %%%
%%%  plunit test suite for Puissance 4. %%%
%%%  Run interactively: tests.          %%%
%%%  CI: swipl -g "run_tests,halt(0)"   %%%
%%%      -t "halt(1)" tests/tests.pl    %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

:- consult('../src/tools/benchmark.pl').    % also consults the game and all the AIs
:- consult('../src/selfplay/selfplay.pl').  % self-play data generator (mcts teacher test)
:- use_module(library(plunit)).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  SHARED TEST BOARDS                 %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% x has three in a row on row 1 (cols 1-3): playing col 4 wins immediately
tb_win_col4(
    [   ['x', 'o', 'o', '_', '_', '_'],
        ['x', 'o', '_', '_', '_', '_'],
        ['x', '_', '_', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_']],
    [4,3,2,1,1,1,1]).

% x wins immediately by playing col 6 (falling diagonal 3,4 / 4,3 / 5,2 / 6,1)
tb_win_col6(
    [   ['_', '_', '_', '_', '_', '_'],
        ['o', '_', '_', '_', '_', '_'],
        ['x', 'o', 'o', 'x', '_', '_'],
        ['o', 'x', 'x', '_', '_', '_'],
        ['o', 'x', '_', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_']],
    [1,2,5,4,3,1,1]).

% o has three stacked in col 2: x must block by playing col 2
tb_block_col2(
    [   ['_', '_', '_', '_', '_', '_'],
        ['o', 'o', 'o', '_', '_', '_'],
        ['x', '_', '_', '_', '_', '_'],
        ['x', '_', '_', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_']],
    [1,4,1,1,1,1,1]).

% o wins by playing col 6 (falling diagonal): x must block col 6
tb_block_col6(
    [   ['_', '_', '_', '_', '_', '_'],
        ['x', '_', '_', '_', '_', '_'],
        ['o', 'x', 'x', 'o', '_', '_'],
        ['x', 'o', 'o', '_', '_', '_'],
        ['x', 'o', '_', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_']],
    [1,2,5,4,3,1,1]).

% Mid-game board with col 6 full and no immediate win for either side
tb_midgame(
    [   ['x', 'o', '_', '_', '_', '_'],
        ['o', 'x', 'o', '_', '_', '_'],
        ['x', 'o', 'x', '_', '_', '_'],
        ['o', '_', '_', '_', '_', '_'],
        ['x', '_', '_', '_', '_', '_'],
        ['o', 'x', 'o', 'x', 'o', 'x'],
        ['o', '_', '_', '_', '_', '_']],
    [3,4,4,2,2,7,2]).

% Col is a valid move: in range and the column is not full
valid_col(Heights, Col) :-
    integer(Col),
    between(1, 7, Col),
    nth1(Col, Heights, H),
    H =< 6.

% Full board with no winner (alternating pairs pattern)
tb_draw_board(
    [   ['x', 'x', 'o', 'o', 'x', 'x'],
        ['o', 'o', 'x', 'x', 'o', 'o'],
        ['x', 'x', 'o', 'o', 'x', 'x'],
        ['o', 'o', 'x', 'x', 'o', 'o'],
        ['x', 'x', 'o', 'o', 'x', 'x'],
        ['o', 'o', 'x', 'x', 'o', 'o'],
        ['x', 'x', 'o', 'o', 'x', 'x']]).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  BOARD UTILITIES                    %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

:- begin_tests(board_utils).

test(replace_first) :-
    replace([a,b,c], 1, x, [x,b,c]).

test(replace_nth) :-
    replace([a,b,c], 2, x, [a,x,c]).

test(set_and_get_cell) :-
    initialize(B, _),
    set_cell(B, 2, 3, 'o', B2),
    get_cell(B2, 2, 3, 'o').

test(get_cell_reads_empty) :-
    initialize(B, _),
    get_cell(B, 4, 5, V),
    empty_mark(E),
    V == E.

test(play_move_gravity) :-
    initialize(B, H),
    play_move(B, H, 1, 3, B2, H2),
    get_cell(B2, 3, 1, 'x'),   % piece lands at row = column height
    nth1(3, H2, 2).            % height is incremented

test(play_move_stacks) :-
    initialize(B, H),
    play_move(B, H, 1, 3, B2, H2),
    play_move(B2, H2, 2, 3, B3, H3),
    get_cell(B3, 3, 1, 'x'),
    get_cell(B3, 3, 2, 'o'),   % second piece stacks on top
    nth1(3, H3, 3).

test(play_move_full_column_fails, [fail]) :-
    initialize(B, _),
    play_move(B, [7,1,1,1,1,1,1], 1, 1, _, _).

test(play_move_out_of_range_fails, [fail]) :-
    initialize(B, H),
    play_move(B, H, 1, 8, _, _).

test(count_identity) :-
    count('x', ['x','o','x','_','x'], 3).

test(count_empty_list) :-
    count('x', [], 0).

:- end_tests(board_utils).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  VICTORY DETECTION                  %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

:- begin_tests(victory).

test(vertical_win) :-
    B = [   ['x', 'x', 'x', 'x', '_', '_'],
            ['_', '_', '_', '_', '_', '_'],
            ['_', '_', '_', '_', '_', '_'],
            ['_', '_', '_', '_', '_', '_'],
            ['_', '_', '_', '_', '_', '_'],
            ['_', '_', '_', '_', '_', '_'],
            ['_', '_', '_', '_', '_', '_']],
    check_victory('x', B),
    \+ check_victory('o', B).

test(horizontal_win) :-
    B = [   ['x', '_', '_', '_', '_', '_'],
            ['x', '_', '_', '_', '_', '_'],
            ['x', '_', '_', '_', '_', '_'],
            ['x', '_', '_', '_', '_', '_'],
            ['_', '_', '_', '_', '_', '_'],
            ['_', '_', '_', '_', '_', '_'],
            ['_', '_', '_', '_', '_', '_']],
    check_victory('x', B).

test(diagonal_up_win) :-
    B = [   ['x', '_', '_', '_', '_', '_'],
            ['_', 'x', '_', '_', '_', '_'],
            ['_', '_', 'x', '_', '_', '_'],
            ['_', '_', '_', 'x', '_', '_'],
            ['_', '_', '_', '_', '_', '_'],
            ['_', '_', '_', '_', '_', '_'],
            ['_', '_', '_', '_', '_', '_']],
    check_victory('x', B).

test(diagonal_down_win) :-
    B = [   ['_', '_', '_', 'x', '_', '_'],
            ['_', '_', 'x', '_', '_', '_'],
            ['_', 'x', '_', '_', '_', '_'],
            ['x', '_', '_', '_', '_', '_'],
            ['_', '_', '_', '_', '_', '_'],
            ['_', '_', '_', '_', '_', '_'],
            ['_', '_', '_', '_', '_', '_']],
    check_victory('x', B).

test(three_in_a_row_is_not_a_win, [fail]) :-
    B = [   ['x', 'x', 'x', '_', '_', '_'],
            ['_', '_', '_', '_', '_', '_'],
            ['_', '_', '_', '_', '_', '_'],
            ['_', '_', '_', '_', '_', '_'],
            ['_', '_', '_', '_', '_', '_'],
            ['_', '_', '_', '_', '_', '_'],
            ['_', '_', '_', '_', '_', '_']],
    check_victory('x', B).

test(empty_board_is_not_a_win) :-
    initialize(B, _),
    \+ check_victory('x', B),
    \+ check_victory('o', B).

% --- Boards ported from the legacy test suite ---

test(legacy_board1_diag_up_x) :-
    B = [   ['_', '_', '_', '_', '_', '_'],
            ['_', 'x', '_', '_', '_', '_'],
            ['_', '_', 'x', '_', '_', '_'],
            ['_', '_', '_', 'x', '_', '_'],
            ['_', '_', '_', '_', 'x', '_'],
            ['_', '_', '_', '_', '_', '_'],
            ['_', '_', '_', '_', '_', '_']],
    check_victory('x', B).

test(legacy_board2_vertical_o) :-
    B = [   ['_', '_', '_', '_', '_', '_'],
            ['_', '_', '_', '_', '_', '_'],
            ['_', 'o', 'o', 'o', 'o', '_'],
            ['_', '_', 'x', '_', '_', '_'],
            ['_', '_', '_', 'x', '_', '_'],
            ['_', '_', '_', '_', 'x', '_'],
            ['_', '_', '_', '_', '_', '_']],
    check_victory('o', B).

test(legacy_board3_vertical_x) :-
    B = [   ['x', '_', '_', '_', '_', '_'],
            ['x', '_', 'o', 'o', '_', '_'],
            ['x', '_', 'x', '_', '_', '_'],
            ['x', '_', '_', 'x', '_', '_'],
            ['_', '_', '_', '_', 'x', '_'],
            ['_', '_', '_', '_', '_', '_'],
            ['_', '_', '_', '_', '_', '_']],
    check_victory('x', B).

test(legacy_board4_diag_down_x) :-
    B = [   ['x', '_', '_', '_', '_', '_'],
            ['_', '_', '_', '_', '_', '_'],
            ['_', '_', 'o', 'o', '_', '_'],
            ['x', '_', 'x', '_', 'x', '_'],
            ['x', '_', '_', 'x', '_', '_'],
            ['_', '_', 'x', '_', 'x', '_'],
            ['_', 'x', '_', '_', '_', '_']],
    check_victory('x', B).

test(legacy_board5_no_winner) :-
    B = [   ['x', '_', '_', '_', '_', '_'],
            ['o', '_', 'o', 'o', '_', '_'],
            ['x', '_', 'x', '_', '_', '_'],
            ['x', '_', '_', 'x', '_', '_'],
            ['_', '_', '_', '_', 'x', '_'],
            ['_', '_', '_', '_', '_', '_'],
            ['_', '_', '_', '_', '_', '_']],
    \+ check_victory('x', B),
    \+ check_victory('o', B).

:- end_tests(victory).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  DRAW DETECTION                     %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

:- begin_tests(draw).

test(full_board_no_winner_is_draw) :-
    tb_draw_board(B),
    \+ check_victory('x', B),
    \+ check_victory('o', B),
    check_draw(B).

test(non_full_board_is_not_draw) :-
    initialize(B, _),
    \+ check_draw(B).

test(draw_heights_full) :-
    check_draw_heights([7,7,7,7,7,7,7]).

test(draw_heights_not_full, [fail]) :-
    check_draw_heights([7,3,7,7,7,7,7]).

:- end_tests(draw).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  WINNING MOVE DETECTION             %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

:- begin_tests(check_winning_move).

test(finds_winning_column) :-
    tb_win_col4(B, H),
    check_winning_move(B, H, 1, 4).

test(finds_only_winning_column) :-
    tb_win_col4(B, H),
    findall(C, check_winning_move(B, H, 1, C), Cs),
    Cs == [4].

test(finds_opponent_winning_column) :-
    tb_block_col2(B, H),
    check_winning_move(B, H, 2, 2).

test(no_winning_move_fails, [fail]) :-
    initialize(B, H),
    check_winning_move(B, H, 1, _).

:- end_tests(check_winning_move).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  RANDOM AI                          %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

:- begin_tests(ai_random).

test(returns_valid_column) :-
    tb_midgame(_, H),
    random_play(H, Col),
    valid_col(H, Col).

test(only_non_full_column_is_chosen) :-
    H = [7,7,7,7,1,7,7],
    random_play(H, Col),
    Col = 5.

:- end_tests(ai_random).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  RANDOM+ AI (attack)                %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

:- begin_tests(ai_random_plus).

% Regression: immediate wins are still taken after the
% check_winning_move merge into board.pl
test(takes_immediate_win_col4) :-
    tb_win_col4(B, H),
    smart_random_play(B, H, 1, Col),
    Col = 4.

test(takes_immediate_win_col6) :-
    tb_win_col6(B, H),
    smart_random_play(B, H, 1, Col),
    Col = 6.

test(fallback_returns_valid_column) :-
    tb_midgame(B, H),
    smart_random_play(B, H, 1, Col),
    valid_col(H, Col).

:- end_tests(ai_random_plus).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  RANDOM++ AI (attack + defense)     %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

:- begin_tests(ai_random_plus_plus).

% Regression: immediate wins are still taken after the
% check_winning_move merge into board.pl
test(takes_immediate_win_col4) :-
    tb_win_col4(B, H),
    with_output_to(atom(_), defensive_random_play(B, H, 1, Col)),
    Col = 4.

test(takes_immediate_win_col6) :-
    tb_win_col6(B, H),
    with_output_to(atom(_), defensive_random_play(B, H, 1, Col)),
    Col = 6.

test(blocks_vertical_threat_col2) :-
    tb_block_col2(B, H),
    with_output_to(atom(_), defensive_random_play(B, H, 1, Col)),
    Col = 2.

test(blocks_diagonal_threat_col6) :-
    tb_block_col6(B, H),
    with_output_to(atom(_), defensive_random_play(B, H, 1, Col)),
    Col = 6.

test(fallback_returns_valid_column) :-
    tb_midgame(B, H),
    with_output_to(atom(_), defensive_random_play(B, H, 1, Col)),
    valid_col(H, Col).

:- end_tests(ai_random_plus_plus).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  MINIMAX AI (transposition table)   %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% clean_all clears the transposition table and the last-turn score
% so games/test cases do not contaminate each other.

:- begin_tests(ai_minimax).

test(takes_immediate_win_col4, [setup(clean_all), cleanup(clean_all)]) :-
    tb_win_col4(B, H),
    with_output_to(atom(_), ia_play(B, H, 'x', Col)),
    Col = 4.

test(takes_immediate_win_col6, [setup(clean_all), cleanup(clean_all)]) :-
    tb_win_col6(B, H),
    with_output_to(atom(_), ia_play(B, H, 'x', Col)),
    Col = 6.

test(blocks_vertical_threat_col2, [setup(clean_all), cleanup(clean_all)]) :-
    tb_block_col2(B, H),
    with_output_to(atom(_), ia_play(B, H, 'x', Col)),
    Col = 2.

test(blocks_diagonal_threat_col6, [setup(clean_all), cleanup(clean_all)]) :-
    tb_block_col6(B, H),
    with_output_to(atom(_), ia_play(B, H, 'x', Col)),
    Col = 6.

test(returns_valid_column, [setup(clean_all), cleanup(clean_all)]) :-
    tb_midgame(B, H),
    with_output_to(atom(_), ia_play(B, H, 'x', Col)),
    valid_col(H, Col).

:- end_tests(ai_minimax).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  NEURAL NETWORK AI (Janus bridge)   %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% nn_clean_all clears the NN transposition table and the Python eval
% cache so test cases do not contaminate each other. The model is
% currently random-init, but terminal scores (+/-100000) dominate any
% leaf eval in [-1000, 1000], so wins/blocks are still found by search.

:- begin_tests(ai_nn).

test(takes_immediate_win_col4, [setup(nn_clean_all), cleanup(nn_clean_all)]) :-
    tb_win_col4(B, H),
    with_output_to(atom(_), nn_ia_play(B, H, 'x', Col)),
    Col = 4.

test(blocks_vertical_threat_col2, [setup(nn_clean_all), cleanup(nn_clean_all)]) :-
    tb_block_col2(B, H),
    with_output_to(atom(_), nn_ia_play(B, H, 'x', Col)),
    Col = 2.

test(blocks_diagonal_threat_col6, [setup(nn_clean_all), cleanup(nn_clean_all)]) :-
    tb_block_col6(B, H),
    with_output_to(atom(_), nn_ia_play(B, H, 'x', Col)),
    Col = 6.

test(returns_valid_column, [setup(nn_clean_all), cleanup(nn_clean_all)]) :-
    tb_midgame(B, H),
    with_output_to(atom(_), nn_ia_play(B, H, 'x', Col)),
    valid_col(H, Col).

% Resilience: nn_eval must return an integer whether the Janus/py_call
% path works or the heuristic_eval fallback kicks in - never throw.
test(nn_eval_returns_integer, [setup(nn_clean_all), cleanup(nn_clean_all)]) :-
    tb_midgame(B, _),
    nn_eval(B, 'x', Score),
    integer(Score).

% Policy-guided move ordering returns a permutation of the valid columns
% (the order itself depends on the current policy head).
test(policy_ordering_returns_permutation, [setup(nn_clean_all), cleanup(nn_clean_all)]) :-
    tb_midgame(B, H),
    nn_get_ordered_moves(B, H, 'x', 3, Ordered),
    ordered_valid_moves(H, Valid),
    msort(Ordered, Sorted),
    msort(Valid, Sorted).

% With policy ordering disabled (min depth 99), the TT-first/center-order
% fallback path also returns a permutation of the valid columns.
test(policy_ordering_fallback_returns_permutation,
     [setup(nn_clean_all), cleanup(nn_clean_all)]) :-
    tb_midgame(B, H),
    nn_policy_order_min_depth(Old),
    setup_call_cleanup(
        set_nn_policy_order_min_depth(99),
        ( nn_get_ordered_moves(B, H, 'x', 3, Ordered),
          ordered_valid_moves(H, Valid),
          msort(Ordered, Sorted),
          msort(Valid, Sorted) ),
        set_nn_policy_order_min_depth(Old)).

:- end_tests(ai_nn).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  ALPHA-BETA AI (pedagogical)        %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

:- begin_tests(ai_alphabeta).

test(takes_immediate_win_col4) :-
    tb_win_col4(B, H),
    once(alpha_beta(B, H, 8, 1, simple_eval, -999999, 999999, Col, _)),
    Col = 4.

% Note: the exact column is not asserted here (this was the commented-out
% assertion in the legacy suite). Besides the immediate win at col 6,
% col 4 is also a forced win at depth 8, and the pedagogical alpha-beta
% returns the first best column in 1..7 order (it returns 4).
test(winning_move_is_valid_col6_board) :-
    tb_win_col6(B, H),
    once(alpha_beta(B, H, 8, 1, simple_eval, -999999, 999999, Col, _)),
    valid_col(H, Col).

test(blocks_vertical_threat_col2) :-
    tb_block_col2(B, H),
    once(alpha_beta(B, H, 8, 1, simple_eval, -999999, 999999, Col, _)),
    Col = 2.

test(blocks_diagonal_threat_col6) :-
    tb_block_col6(B, H),
    once(alpha_beta(B, H, 8, 1, simple_eval, -999999, 999999, Col, _)),
    Col = 6.

test(returns_valid_column) :-
    tb_midgame(B, H),
    once(alpha_beta(B, H, 8, 1, simple_eval, -999999, 999999, Col, _)),
    valid_col(H, Col).

:- end_tests(ai_alphabeta).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  MCTS AI                            %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

:- begin_tests(ai_mcts).

test(returns_valid_column) :-
    tb_midgame(B, H),
    with_output_to(atom(_), mcts_play(B, H, 1, Col)),
    valid_col(H, Col).

:- end_tests(ai_mcts).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  BENCHMARK SMOKE TEST               %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

:- begin_tests(benchmark_smoke).

test(random_vs_random_completes,
     [cleanup((retractall(board(_)),
               retractall(heights(_)),
               retractall(player_type(_, _))))]) :-
    with_output_to(atom(_),
        benchmark(computer_random, computer_random, 2, WA, WB, D)),
    Total is WA + WB + D,
    Total =:= 2.

:- end_tests(benchmark_smoke).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  SELF-PLAY DATA GENERATION          %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

:- begin_tests(selfplay).

% Smoke test: two short MCTS-teacher games produce a valid data file.
% (The path is relative to the working directory: run from the project root.)
test(mcts_teacher_smoke,
     [cleanup(catch(delete_file('data/test_smoke.txt'), _, true))]) :-
    gen_games(2, 'data/test_smoke.txt', spec((mcts,100), (mcts,100), 0.1, 4)),
    exists_file('data/test_smoke.txt'),
    read_file_to_string('data/test_smoke.txt', Content, []),
    split_string(Content, "\n", " \t\r", RawLines),
    exclude(==(""), RawLines, Lines),       % drop the trailing empty piece
    Lines = [_|_],                          % non-empty file
    forall(member(Line, Lines), valid_data_line(Line)).

% One data line: "<board42> <x|o> <1|-1|0> <move>" where move is the
% AI-chosen column 1-7, or '-' for random (opening/exploration) moves
valid_data_line(Line) :-
    split_string(Line, " ", " ", [Board, Player, Result, Move]),
    string_length(Board, 42),
    member(Player, ["x", "o"]),
    member(Result, ["1", "-1", "0"]),
    member(Move, ["-", "1", "2", "3", "4", "5", "6", "7"]).

:- end_tests(selfplay).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  CONVENIENCE ALIAS                  %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Legacy entry point: "tests." still runs the whole suite
tests :-
    run_tests.
