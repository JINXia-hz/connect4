%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  src/core/board.pl                  %%%
%%%  Shared game constants and board    %%%
%%%  utilities for Puissance 4.         %%%
%%%  Main predicates: num_rows/1,       %%%
%%%  num_cols/1, empty_mark/1,          %%%
%%%  player_mark/2, next_player/2,      %%%
%%%  replace/4, set_cell/5,             %%%
%%%  play_move/6, get_element/3,        %%%
%%%  get_cell/4, count/3,               %%%
%%%  check_winning_move/4               %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Board dimensions
num_rows(6).
num_cols(7).

% Cell markers
empty_mark('_').
player_mark(1, 'x').
player_mark(2, 'o').

% Turn order (players are numbered 1 and 2)
next_player(1, 2).
next_player(2, 1).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%     LIST UTILITIES                  %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Replace the I-th element of a list (1-based)
replace([_|T], 1, X, [X|T]).
replace([H|T], I, X, [H|R]) :-
    I > 1,
    I2 is I - 1,
    replace(T, I2, X, R).

% Get the N-th element of a list (1-based)
get_element([H|_], 1, H) :-
    !.
get_element([_|T], N, Res) :-
    N > 1,
    N1 is N - 1,
    get_element(T, N1, Res).

% Get the value at column C, row R of the board
get_cell(Board, C, R, Val) :-
    get_element(Board, C, TargetColumn),
    get_element(TargetColumn, R, Val).

% Count occurrences of Elem in List (identity comparison ==)
count(Elem, List, Total) :-
    count_acc(Elem, List, 0, Total).

count_acc(_, [], Acc, Acc) :- !.
count_acc(Elem, [H|T], Acc, Total) :-
    Elem == H,
    !,
    NewAcc is Acc + 1,
    count_acc(Elem, T, NewAcc, Total).
count_acc(Elem, [_|T], Acc, Total) :-
    count_acc(Elem, T, Acc, Total).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%     BOARD MANIPULATION              %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Set Board[Col][Row] = Value
set_cell(Board, Col, Row, Value, NewBoard) :-
    nth1(Col, Board, OldColumn),
    replace(OldColumn, Row, Value, NewColumn),
    replace(Board, Col, NewColumn, NewBoard).

% Drop Player's piece (Player is 1 or 2) into column Col
play_move(Board, Heights, Player, Col, NewBoard, NewHeights) :-
    num_rows(MaxRow),
    num_cols(MaxCol),
    Col >= 1, Col =< MaxCol,
    nth1(Col, Heights, H),
    H =< MaxRow,
    player_mark(Player, Mark),
    set_cell(Board, Col, H, Mark, NewBoard),
    H2 is H + 1,
    replace(Heights, Col, H2, NewHeights).

% True if Player (1 or 2) wins immediately by playing column Col.
% Requires check_victory/2 (src/core/victory.pl) at run time.
check_winning_move(Board, Heights, Player, Col) :-
    num_cols(MaxCol),
    num_rows(MaxRow),
    between(1, MaxCol, Col),
    nth1(Col, Heights, H),
    H =< MaxRow,
    play_move(Board, Heights, Player, Col, TempBoard, _),
    player_mark(Player, Mark),
    check_victory(Mark, TempBoard).
