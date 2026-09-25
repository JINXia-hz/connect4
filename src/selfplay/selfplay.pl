%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  src/selfplay/selfplay.pl           %%%
%%%  Self-play data generation for the  %%%
%%%  neural-network AI. Plays games     %%%
%%%  between heuristic, NN and/or MCTS  %%%
%%%  agents and writes one training     %%%
%%%  line per recorded position. Also   %%%
%%%  provides recording-free evaluation %%%
%%%  matches (play_match/7).            %%%
%%%  Main predicates: gen_games/3,      %%%
%%%  play_match/7                       %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

:- use_module(library(random)).
:- consult('../core/board.pl').
:- consult('../core/victory.pl').
:- consult('../ai/minimax.pl').
:- consult('../ai/nn.pl').
:- consult('../ai/mcts.pl').

% gen_games(N, OutFile, spec((K1,D1), (K2,D2), Epsilon, OpenRandom))
% Play N games: player x uses spec (K1,D1), player o uses (K2,D2).
% K is 'heuristic' (minimax.pl's alpha_beta/8), 'nn' (nn_alpha_beta/8),
% 'mcts' (MCTS with D iterations per move, used as a teacher so the
% NN never learns from the hand-written heuristic) or 'original'
% (full-strength adaptive minimax ia_play/4; D ignored, use 0).
% Epsilon is the random-move probability (exploration); the first
% OpenRandom plies of each game are fully random.
gen_games(N, OutFile, Spec) :-
    Spec = spec(_, _, _, _),
    file_directory_name(OutFile, Dir),
    make_directory_path(Dir),
    open(OutFile, write, Stream),
    % once/1 prunes leftover choicepoints from the game simulation so the
    % cleanup (close/flush of the output file) runs immediately, not at
    % process exit -- callers in the same process must see complete data.
    call_cleanup(once(gen_games_loop(1, N, Spec, Stream)), close(Stream)).

gen_games_loop(I, N, Spec, Stream) :-
    I =< N, !,
    play_one_game(Spec, Stream, Winner),
    % Reset both transposition tables and the Python eval cache between games
    clean_all,
    nn_clean_all,
    (I mod 10 =:= 0 ->
        format('game ~w/~w done (winner: ~w)~n', [I, N, Winner]),
        flush_output
    ;
        true
    ),
    I1 is I + 1,
    gen_games_loop(I1, N, Spec, Stream).
gen_games_loop(_, _, _, _).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%     EVALUATION MATCHES              %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% play_match(Spec1, Spec2, N, OpenRandom, Wins1, Wins2, Draws)
% Play N games WITHOUT recording positions: Spec1 and Spec2 are (Kind,Param)
% pairs; game 1 has Spec1 as x, game 2 Spec2 as x, alternating. Epsilon is
% 0; the first OpenRandom plies of each game are uniform random. Ends by
% printing a machine-readable "MATCH_RESULT W1 W2 D" line.
play_match(Spec1, Spec2, N, OpenRandom, Wins1, Wins2, Draws) :-
    once(play_match_loop(1, N, Spec1, Spec2, OpenRandom, 0, 0, 0, Wins1, Wins2, Draws)),
    format('MATCH_RESULT ~w ~w ~w~n', [Wins1, Wins2, Draws]),
    flush_output.

play_match_loop(I, N, _, _, _, W1, W2, D, W1, W2, D) :-
    I > N, !.
play_match_loop(I, N, Spec1, Spec2, OpenR, WA0, WB0, D0, WA, WB, D) :-
    % Alternate sides: odd games Spec1 is x, even games Spec2 is x
    (I mod 2 =:= 1 ->
        GameSpec = spec(Spec1, Spec2, 0.0, OpenR), Spec1IsX = true
    ;
        GameSpec = spec(Spec2, Spec1, 0.0, OpenR), Spec1IsX = false
    ),
    empty_state(Board, Heights),
    play_game(Board, Heights, 'x', 1, GameSpec, false, [], _, Winner),
    % Reset both transposition tables and the Python eval cache between games
    clean_all,
    nn_clean_all,
    winner_index(Winner, Spec1IsX, W),
    (W == 1 -> WA1 is WA0 + 1, WB1 = WB0, D1 = D0
    ; W == 2 -> WB1 is WB0 + 1, WA1 = WA0, D1 = D0
    ; D1 is D0 + 1, WA1 = WA0, WB1 = WB0),
    format('match game ~w/~w: winner ~w~n', [I, N, W]),
    flush_output,
    I1 is I + 1,
    play_match_loop(I1, N, Spec1, Spec2, OpenR, WA1, WB1, D1, WA, WB, D).

% Map the winning mark to the winning spec index (1, 2 or draw)
winner_index(draw, _, draw) :- !.
winner_index('x', true, 1) :- !.
winner_index('o', true, 2) :- !.
winner_index('x', false, 2) :- !.
winner_index('o', false, 1).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%     ONE GAME                        %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Play a single game and write the recorded positions to Stream.
play_one_game(Spec, Stream, Winner) :-
    empty_state(Board, Heights),
    play_game(Board, Heights, 'x', 1, Spec, [], Positions, Winner),
    write_positions(Stream, Positions, Winner).

% Empty initial state (built directly, without puissance4.pl's game machinery)
empty_state(Board, Heights) :-
    Col = ['_', '_', '_', '_', '_', '_'],
    Board = [Col, Col, Col, Col, Col, Col, Col],
    Heights = [1, 1, 1, 1, 1, 1, 1].

% Game loop: Mark is the side to move, Ply the current ply number.
% Every position is recorded (side to move + board + chosen move);
% the move is '-' when it came from the random opening, from
% epsilon-exploration, or from a hand-written-rules agent
% (heuristic / original): those must NOT become policy labels.
play_game(Board, Heights, Mark, Ply, Spec, Acc, Positions, Winner) :-
    play_game(Board, Heights, Mark, Ply, Spec, true, Acc, Positions, Winner).

% play_game(+Board, +Heights, +Mark, +Ply, +Spec, +Record, +Acc, -Positions, -Winner)
% Record = true: accumulate positions; Record = false: play without
% recording (evaluation matches).
play_game(Board, Heights, Mark, Ply, Spec, Record, Acc, Positions, Winner) :-
    (game_end(Board, Winner) ->
        Positions = Acc
    ;
        choose_move(Spec, Mark, Ply, Board, Heights, Col, Source),
        (Record == true ->
            (Source == ai -> MoveField = Col ; MoveField = '-'),
            Acc1 = [pos(Board, Mark, MoveField) | Acc]
        ;
            Acc1 = Acc
        ),
        simulate_move(Board, Heights, Mark, Col, NewBoard, NewHeights),
        ai_change_player(Mark, NextMark),
        Ply1 is Ply + 1,
        play_game(NewBoard, NewHeights, NextMark, Ply1, Spec, Record, Acc1, Positions, Winner)
    ).

% Winner is 'x', 'o' or 'draw'
game_end(Board, 'x') :-
    check_victory('x', Board), !.
game_end(Board, 'o') :-
    check_victory('o', Board), !.
game_end(Board, draw) :-
    ai_check_draw(Board).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%     MOVE SELECTION                  %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Opening plies are uniform random; after that, move randomly with
% probability Epsilon (exploration), otherwise let the AI play.
% Source is 'ai' when the column was chosen by an AI whose moves may
% become policy labels (nn, mcts), 'unlabeled_ai' for hand-written-rules
% agents (heuristic, original), 'random' for random moves.
choose_move(Spec, Mark, Ply, Board, Heights, Col, Source) :-
    Spec = spec(P1, P2, Eps, OpenR),
    (Mark == 'x' -> PlayerSpec = P1 ; PlayerSpec = P2),
    (Ply =< OpenR ->
        random_valid_move(Heights, Col), Source = random
    ;
        random(R),
        R < Eps
    ->
        random_valid_move(Heights, Col), Source = random
    ;
        ai_move(PlayerSpec, Board, Heights, Mark, Col, Source)
    ).

% Heuristic AI: fixed-depth alpha-beta from minimax.pl. Hand-written
% rules must not become policy labels -> Source = unlabeled_ai.
ai_move((heuristic, D), Board, Heights, Mark, Col, unlabeled_ai) :-
    catch(alpha_beta(Board, Heights, Mark, D, -9999999, 9999999, Col0, _), _, fail),
    valid_column(Heights, Col0), !,
    Col = Col0.

% Original full-strength minimax (adaptive depth + panic + quiescence,
% ia_play/4): the fixed reference ruler. The numeric parameter is accepted
% for spec uniformity and ignored. Hand-written rules must not become
% policy labels -> Source = unlabeled_ai.
ai_move((original, _), Board, Heights, Mark, Col, unlabeled_ai) :-
    catch(with_output_to(atom(_), ia_play(Board, Heights, Mark, Col0)), _, fail),
    valid_column(Heights, Col0), !,
    Col = Col0.

% NN AI: fixed-depth alpha-beta with neural evaluation
ai_move((nn, D), Board, Heights, Mark, Col, ai) :-
    catch(nn_alpha_beta(Board, Heights, Mark, D, -9999999, 9999999, Col0, _), _, fail),
    valid_column(Heights, Col0), !,
    Col = Col0.

% MCTS teacher: D is the iteration count. mcts_play/4 uses player
% numbers (x=1, o=2), so the mark-based state is converted. Its chatter
% is suppressed; it cleans its own dynamic state on every call.
ai_move((mcts, Iters), Board, Heights, Mark, Col, ai) :-
    mark_to_number(Mark, PlayerNum),
    set_mcts_iterations(Iters),
    catch(with_output_to(atom(_), mcts_play(Board, Heights, PlayerNum, Col0)), _, fail),
    valid_column(Heights, Col0), !,
    Col = Col0.

% Robustness: if an AI call fails or returns an invalid column,
% fall back to a random valid move instead of crashing the batch.
% Such a move is not usable as a policy target (Source = random).
ai_move(_, _, Heights, _, Col, random) :-
    random_valid_move(Heights, Col).

mark_to_number('x', 1).
mark_to_number('o', 2).

% Uniform random among the non-full columns
random_valid_move(Heights, Col) :-
    ordered_valid_moves(Heights, Moves),
    random_member(Col, Moves).

valid_column(Heights, Col) :-
    integer(Col),
    between(1, 7, Col),
    nth1(Col, Heights, H),
    H =< 6.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%     OUTPUT                          %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% One line per recorded position: "<board42> <player> <result> <move>"
% (contract with ml/encode.py: column-major, cols 1..7, rows 1..6 bottom-up;
% result is from the recorded side-to-move's perspective: 1/-1/0;
% move is the chosen column 1..7 for nn/mcts agents, or '-' when the move
% was random (opening / epsilon exploration / AI fallback) or chosen by a
% hand-written-rules agent (heuristic / original) - not a policy target)
write_positions(Stream, Positions, Winner) :-
    forall(member(pos(Board, Mark, Move), Positions),
           (result_for(Mark, Winner, Result),
            board_to_str42(Board, Str),
            format(Stream, '~w ~w ~w ~w~n', [Str, Mark, Result, Move]))).

result_for(_, draw, 0) :- !.
result_for(Mark, Winner, 1) :- Mark == Winner, !.
result_for(_, _, -1).

% Board (list of 7 columns of 6 cells, bottom-up) -> 42-char atom,
% column-major: column 1 rows 1..6, then column 2, etc.
board_to_str42(Board, Str) :-
    append(Board, Cells),
    atomic_list_concat(Cells, '', Str).
