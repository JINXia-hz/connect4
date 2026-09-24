%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  src/ai/alphabeta.pl                %%%
%%%  Pedagogical alpha-beta minimax     %%%
%%%  (players are numbers 1/2).         %%%
%%%  Main predicates: alpha_beta/9,     %%%
%%%  valid_moves/2                      %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

:- consult('../core/board.pl').
:- consult('../core/victory.pl').

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  TERMINAL POSITIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Score of a terminal position from Player's perspective
terminal(Board, Heights, Player, Score) :-
    next_player(Player, Opp),
    ( check_victory_player(Board, Player) -> Score = 10000
    ; check_victory_player(Board, Opp) -> Score = -10000
    ; check_draw_heights(Heights) -> Score = 0
    ; fail ).

check_victory_player(Board, Player) :-
    player_mark(Player, Mark),
    check_victory(Mark, Board).

check_draw_heights(Heights) :-
    num_rows(Max),
    \+ ( member(H, Heights), H =< Max ).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  VALID MOVES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Non-full columns; fails when the board is full
valid_moves(Heights, Moves) :-
    num_cols(N),
    num_rows(Max),
    findall(
        Col,
        ( between(1, N, Col),
          nth1(Col, Heights, H),
          H =< Max ),
        Moves
    ),
    Moves \= [].

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  ENTRY POINT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

alpha_beta(Board, Heights, Depth, Player, _EvalFunc, Alpha, Beta, BestCol, BestScore) :-
    valid_moves(Heights, Moves),
    best_move(Moves, Board, Heights, Depth, Player, Alpha, Beta, nil, -100000, BestCol, BestScore).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  MAIN LOOP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

best_move([], _, _, _, _, _, _, BestCol, BestScore, BestCol, BestScore).

best_move([Col|Rest], Board, Heights, Depth, Player, Alpha, Beta, CurCol, CurScore, BestCol, BestScore) :-
    play_move(Board, Heights, Player, Col, NewBoard, NewHeights),
    ( terminal(NewBoard, NewHeights, Player, Score) ->
        true
    ; Depth =< 1 ->
        Score = 0
    ; next_player(Player, Opp),
      D1 is Depth - 1,
      alpha_beta(NewBoard, NewHeights, D1, Opp, _, -Beta, -Alpha, _, Score1),
      Score is -Score1
    ),
    ( Score > CurScore ->
        Alpha1 is max(Alpha, Score),
        ( Alpha1 >= Beta ->
            BestCol = Col,
            BestScore = Score
        ; best_move(Rest, Board, Heights, Depth, Player, Alpha1, Beta, Col, Score, BestCol, BestScore)
        )
    ;
        best_move(Rest, Board, Heights, Depth, Player, Alpha, Beta, CurCol, CurScore, BestCol, BestScore)
    ).
