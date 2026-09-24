%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  src/selfplay/selfplay.pl           %%%
%%%  Self-play data generation for the  %%%
%%%  neural-network AI. Plays games     %%%
%%%  between heuristic, NN and/or MCTS  %%%
%%%  agents and writes one training     %%%
%%%  line per recorded position.        %%%
%%%  Main predicates: gen_games/3       %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

:- use_module(library(random)).
:- consult('../core/board.pl').
:- consult('../core/victory.pl').
:- consult('../ai/minimax.pl').
:- consult('../ai/nn.pl').
:- consult('../ai/mcts.pl').

% gen_games(N, OutFile, spec((K1,D1), (K2,D2), Epsilon, OpenRandom))
% Play N games: player x uses spec (K1,D1), player o uses (K2,D2).
% K is 'heuristic' (minimax.pl's alpha_beta/8), 'nn' (nn_alpha_beta/8)
% or 'mcts' (MCTS with D iterations per move, used as a teacher so the
% NN never learns from the hand-written heuristic).
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
% Positions are accumulated (side to move + board) from ply OpenRandom+1 on.
play_game(Board, Heights, Mark, Ply, Spec, Acc, Positions, Winner) :-
    (game_end(Board, Winner) ->
        Positions = Acc
    ;
        Spec = spec(_, _, _, OpenR),
        (Ply > OpenR ->
            Acc1 = [pos(Board, Mark) | Acc]
        ;
            Acc1 = Acc
        ),
        choose_move(Spec, Mark, Ply, Board, Heights, Col),
        simulate_move(Board, Heights, Mark, Col, NewBoard, NewHeights),
        ai_change_player(Mark, NextMark),
        Ply1 is Ply + 1,
        play_game(NewBoard, NewHeights, NextMark, Ply1, Spec, Acc1, Positions, Winner)
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
choose_move(Spec, Mark, Ply, Board, Heights, Col) :-
    Spec = spec(P1, P2, Eps, OpenR),
    (Mark == 'x' -> PlayerSpec = P1 ; PlayerSpec = P2),
    (Ply =< OpenR ->
        random_valid_move(Heights, Col)
    ;
        random(R),
        R < Eps
    ->
        random_valid_move(Heights, Col)
    ;
        ai_move(PlayerSpec, Board, Heights, Mark, Col)
    ).

% Heuristic AI: fixed-depth alpha-beta from minimax.pl
ai_move((heuristic, D), Board, Heights, Mark, Col) :-
    catch(alpha_beta(Board, Heights, Mark, D, -9999999, 9999999, Col0, _), _, fail),
    valid_column(Heights, Col0), !,
    Col = Col0.

% NN AI: fixed-depth alpha-beta with neural evaluation
ai_move((nn, D), Board, Heights, Mark, Col) :-
    catch(nn_alpha_beta(Board, Heights, Mark, D, -9999999, 9999999, Col0, _), _, fail),
    valid_column(Heights, Col0), !,
    Col = Col0.

% MCTS teacher: D is the iteration count. mcts_play/4 uses player
% numbers (x=1, o=2), so the mark-based state is converted. Its chatter
% is suppressed; it cleans its own dynamic state on every call.
ai_move((mcts, Iters), Board, Heights, Mark, Col) :-
    mark_to_number(Mark, PlayerNum),
    set_mcts_iterations(Iters),
    catch(with_output_to(atom(_), mcts_play(Board, Heights, PlayerNum, Col0)), _, fail),
    valid_column(Heights, Col0), !,
    Col = Col0.

% Robustness: if an AI call fails or returns an invalid column,
% fall back to a random valid move instead of crashing the batch.
ai_move(_, _, Heights, _, Col) :-
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

% One line per recorded position: "<board42> <player> <result>"
% (contract with ml/encode.py: column-major, cols 1..7, rows 1..6 bottom-up;
% result is from the recorded side-to-move's perspective: 1/-1/0)
write_positions(Stream, Positions, Winner) :-
    forall(member(pos(Board, Mark), Positions),
           (result_for(Mark, Winner, Result),
            board_to_str42(Board, Str),
            format(Stream, '~w ~w ~w~n', [Str, Mark, Result]))).

result_for(_, draw, 0) :- !.
result_for(Mark, Winner, 1) :- Mark == Winner, !.
result_for(_, _, -1).

% Board (list of 7 columns of 6 cells, bottom-up) -> 42-char atom,
% column-major: column 1 rows 1..6, then column 2, etc.
board_to_str42(Board, Str) :-
    append(Board, Cells),
    atomic_list_concat(Cells, '', Str).
