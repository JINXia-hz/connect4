%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  src/tools/benchmark.pl             %%%
%%%  Benchmark harness: plays N games   %%%
%%%  between two AIs, alternating who   %%%
%%%  starts, and tallies the results.   %%%
%%%  Main predicates: benchmark/6       %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

:- consult('../game/puissance4.pl').

% Entry point
benchmark(AlgoA, AlgoB, N, WinsA, WinsB, Draws) :-
    benchmark_(1, N, AlgoA, AlgoB, 0, 0, 0, WinsA, WinsB, Draws).

% Main loop (counts wins/losses/draws)
benchmark_(I, N, _, _, WA, WB, D, WA, WB, D) :-
    I > N, !.
benchmark_(I, N, A, B, WA0, WB0, D0, WA, WB, D) :-
    play_fair_game(A, B, R),
    ( R = 'x' -> WA1 is WA0 + 1, WB1 = WB0,     D1 = D0
    ; R = 'o' -> WB1 is WB0 + 1, WA1 = WA0,     D1 = D0
    ; R = '_'   -> D1 is D0 + 1,   WA1 = WA0,     WB1 = WB0
    ),
    I1 is I + 1,
    benchmark_(I1, N, A, B, WA1, WB1, D1, WA, WB, D).

% Randomly decide who plays first in each game
play_fair_game(A, B, Result) :-
    ( random(0,2,0) ->
        setup_game(A, B),
        start_game(Result)
    ;   setup_game(B, A),
        start_game(R0),
        invert_result(R0, Result)
    ).

invert_result('x', 'o').
invert_result('o', 'x').
invert_result('_', '_').