%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  src/core/victory.pl                %%%
%%%  Win and draw detection.            %%%
%%%  Main predicates: check_victory/2,  %%%
%%%  check_draw/1                       %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

:- consult('board.pl').

% A player (mark 'x' or 'o') wins vertically, horizontally or diagonally
check_victory(Player, Board) :-
    check_vertical_victory(Player, Board).
check_victory(Player, Board) :-
    check_horizontal_victory(Player, Board).
check_victory(Player, Board) :-
    win_diag_up(Board, Player).
check_victory(Player, Board) :-
    win_diag_down(Board, Player).

% --- Vertical victory ---
check_vertical_victory(Player, Board) :-
    member(Column, Board),
    four_in_column(Player, Column).

% Four identical non-empty marks in a column
four_in_column(Player, [Player,Player,Player,Player|_]) :-
    Player \= '_'.
four_in_column(Player, [_|T]) :-
    four_in_column(Player, T).

% --- Draw ---

% A column is full when it contains no empty cell
column_full(Column) :-
    empty_mark(Empty),
    \+ member(Empty, Column).

board_full([]).
board_full([Column|Rest]) :-
    column_full(Column),
    board_full(Rest).

check_draw(Board) :-
    board_full(Board),
    !.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%     DIAGONAL VICTORIES              %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% --- Rising diagonal (/): (C,R), (C+1,R+1), (C+2,R+2), (C+3,R+3) ---
win_diag_up(Board, Mark) :-
    between(1, 4, C),
    between(1, 3, R),

    get_cell(Board, C, R, Mark),

    C2 is C + 1, R2 is R + 1,
    get_cell(Board, C2, R2, Mark),

    C3 is C + 2, R3 is R + 2,
    get_cell(Board, C3, R3, Mark),

    C4 is C + 3, R4 is R + 3,
    get_cell(Board, C4, R4, Mark).

% --- Falling diagonal (\): (C,R), (C+1,R-1), (C+2,R-2), (C+3,R-3) ---
win_diag_down(Board, Mark) :-
    between(1, 4, C),
    between(4, 6, R),

    get_cell(Board, C, R, Mark),

    C2 is C + 1, R2 is R - 1,
    get_cell(Board, C2, R2, Mark),

    C3 is C + 2, R3 is R - 2,
    get_cell(Board, C3, R3, Mark),

    C4 is C + 3, R4 is R - 3,
    get_cell(Board, C4, R4, Mark).

% --- Horizontal victory ---
check_horizontal_victory(Player, Board) :-
    between(1, 6, R),
    between(1, 4, C),

    get_cell(Board, C, R, Player),

    C2 is C + 1,
    get_cell(Board, C2, R, Player),

    C3 is C + 2,
    get_cell(Board, C3, R, Player),

    C4 is C + 3,
    get_cell(Board, C4, R, Player).
