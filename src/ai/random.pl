%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  src/ai/random.pl                   %%%
%%%  Random-family AIs (merged):        %%%
%%%  - Random:   uniform random move    %%%
%%%  - Random+:  takes immediate wins   %%%
%%%  - Random++: wins, blocks, else     %%%
%%%    random                           %%%
%%%  Main predicates: random_play/2,    %%%
%%%  smart_random_play/4,               %%%
%%%  defensive_random_play/4            %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

:- use_module(library(random)).
:- consult('../core/board.pl').

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  RANDOM                             %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Choose a random column among those that are not full
random_play(Heights, Col) :-
    findall(Index,
            (nth1(Index, Heights, H),
             H =< 6),
            ValidCols),
    random_member(Col, ValidCols).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  RANDOM+ (attack)                   %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Rule 1: take an immediate winning move
smart_random_play(Board, Heights, Player, Col) :-
    check_winning_move(Board, Heights, Player, Col),
    !.

% Rule 2: otherwise fall back to a random move
smart_random_play(_, Heights, _, Col) :-
    random_play(Heights, Col).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  RANDOM++ (attack + defense)        %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Rule 1: take an immediate winning move
defensive_random_play(Board, Heights, Player, Col) :-
    check_winning_move(Board, Heights, Player, Col),
    write(' (Attaque : Coup gagnant !) '),
    !.

% Rule 2: block the opponent's immediate winning move
defensive_random_play(Board, Heights, Player, Col) :-
    next_player(Player, Opponent),
    check_winning_move(Board, Heights, Opponent, Col),
    write(' (Defense : Blocage !) '),
    !.

% Rule 3: otherwise play randomly
defensive_random_play(_, Heights, _, Col) :-
    random_play(Heights, Col).
