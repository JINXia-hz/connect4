%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  src/ai/minimax.pl                  %%%
%%%  Advanced alpha-beta AI with a      %%%
%%%  transposition table and adaptive   %%%
%%%  depth (players are marks 'x'/'o'). %%%
%%%  Main predicates: ia_play/4,        %%%
%%%  alpha_beta/8                       %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

:- use_module(library(lists)).
:- use_module(library(random)).
:- consult('../core/board.pl').
:- encoding(utf8).
% ==============================================================================
%    MEMORY & CACHING (Transposition Table & Pre-computation)
% ==============================================================================

% Stores the evaluation score of the last move to decide if the AI should "panic"
% (think deeper) in the next turn.
:- dynamic last_turn_score/1.
clean_last_score :-
    retractall(last_turn_score(_)).

% Transposition Table (TT): Stores board states we have already analyzed.
% Format: memory_table(Hash, Board, Depth, Flag, Score, BestMove).
% This prevents re-calculating the same position multiple times.
:- dynamic memory_table/6.

clean_cache :-
    retractall(memory_table(_, _, _, _, _, _)).

clean_all :-
    clean_cache,
    retractall(last_turn_score(_)).

% Caches all possible winning combinations (4-cell windows) to speed up evaluation.
:- dynamic cached_windows/1.
:- initialization(precompute_windows).

precompute_windows :-
    (cached_windows(_) -> true ;
        findall(Coords, generate_window_coords(Coords), All),
        assertz(cached_windows(All))
    ).

% Generates coordinates for: Horizontal, Vertical, and both Diagonal lines.
generate_window_coords([ (C,R), (C1,R), (C2,R), (C3,R) ]) :- 
    between(1, 6, R), between(1, 4, C), C1 is C+1, C2 is C+2, C3 is C+3.
generate_window_coords([ (C,R), (C,R1), (C,R2), (C,R3) ]) :- 
    between(1, 3, R), between(1, 7, C), R1 is R+1, R2 is R+2, R3 is R+3.
generate_window_coords([ (C,R), (C1,R1), (C2,R2), (C3,R3) ]) :- 
    between(1, 3, R), between(1, 4, C), C1 is C+1, R1 is R+1, C2 is C+2, R2 is R+2, C3 is C+3, R3 is R+3.
generate_window_coords([ (C,R), (C1,R1), (C2,R2), (C3,R3) ]) :- 
    between(4, 6, R), between(1, 4, C), C1 is C+1, R1 is R-1, C2 is C+2, R2 is R-2, C3 is C+3, R3 is R-3.


% ==============================================================================
%    AI PLAY INTERFACE
% ==============================================================================

% Main entry point for the AI to make a move.
% 1. Cleans old cache.
% 2. Calculates dynamic depth (how many moves ahead to look).
% 3. Runs Alpha-Beta search.
ia_play(Board, Heights, Player, BestCol) :-
    %clean_cache,   % Clearing the TT every move = amnesia (slower). Uncomment for repeated benchmark runs.
    decide_depth(Board, Heights, Player, Depth),
    write('>>> AI Move: '), write(Player), 
    write(' | Depth: '), write(Depth), nl,
    
    % Start Alpha-Beta with infinite window (-9999999 to +9999999)
    (alpha_beta(Board, Heights, Player, Depth, -9999999, 9999999, BestCol, Score) ->
        write('>>> Decision: Col '), write(BestCol), write(' (Score '), write(Score), write(')'), nl,
        retractall(last_turn_score(_)),
        assertz(last_turn_score(Score))
    ;
        write('>>> AI Fail! Fallback random.'), nl, fail
    ).

% Adaptive Depth Logic:
% - Early game: Depth 6.
% - Mid game: Depth 8.
% - Late game: Depth 10.
% - Panic Mode: If the previous move had a very bad score, search 2 levels deeper.
decide_depth(Board, Heights, Player, Depth) :-
    sum_list(Heights, Sum),
    PiecesPlayed is Sum - 7, % Calculate how many pieces are on board
    
    (PiecesPlayed < 10 -> BaseDepth = 5
    ; PiecesPlayed < 24 -> BaseDepth = 6
    ; BaseDepth = 7),

    (last_turn_score(MemScore) -> 
        true 
    ; 
        MemScore = 0 
    ),

    (MemScore \= 0 -> 
        write('>>> Memory says: '), write(MemScore), nl
    ; true),
    adjust_depth_based_on_score(MemScore, BaseDepth, Depth).

adjust_depth_based_on_score(Score, Base, Final) :-
    Score < -100, !, % Threshold for "Panic"
    Final is Base + 2,
    write('>>> Status: PANIC! (Opponent is strong, thinking hard...)'), nl.

adjust_depth_based_on_score(_, Base, Base) :-
    write('>>> Status: FOCUSED. (Standard think)'), nl.


% ==============================================================================
%    STABLE-CHECK (Quiescence / Search Extension)
% ==============================================================================

% Checks if the board is "unstable" (i.e., someone is about to win).
% Used to extend the search depth slightly if a critical situation is detected at depth 0.
not_stable(Board, Heights) :-
    cached_windows(AllCoords),
    member(WindowCoords, AllCoords),
    check_window_stablity(WindowCoords, Board, Heights).

% Checks if a specific 4-cell window has 3 pieces and a playable empty spot (winning threat).
check_window_stablity(WindowCoords, Board, Heights) :-
    extract_values(Board, WindowCoords, Cells),
    member('_', Cells), 
    (count('x', Cells, 3) ; count('o', Cells, 3)), % 3 Xs or 3 Os
    nth1(Index, Cells, '_'),
    nth1(Index, WindowCoords, (Col, Row)),
    nth1(Col, Heights, H),
    H =:= Row. % Checks if the empty spot is immediately playable (gravity).

% ==============================================================================
%    MINIMAX WITH ALPHA-BETA PRUNING
% ==============================================================================

% 1. Transposition Table Lookup
% If we have seen this board state before at a sufficient depth, return the cached result.
alpha_beta(Board, _, _, Depth, Alpha, Beta, BestMove, BestScore) :-
    memory_lookup(Board, Depth, Alpha, Beta, CachedScore, CachedMove),
    !,
    BestScore = CachedScore,
    BestMove = CachedMove.

% 2. Hard Depth Limit
% If we went too deep (extended search), stop immediately to prevent stack overflow/timeout.
alpha_beta(Board, Heights, Player, Depth, _, _, -1, Score) :-
    Depth =< -2,
    !, 
    heuristic_eval(Board, Heights, Player, Score).

% 3. Search Extension (Quiescence-like)
% If Depth is 0 but the board is "not stable" (immediate threat exists),
% extend the search by 1 level (NewDepth = -1).
alpha_beta(Board, Heights, Player, Depth, Alpha, Beta, BestMove, BestScore) :-
    Depth =< 0,
    not_stable(Board, Heights),
    !, 
    NewDepth is Depth - 1,
    get_ordered_moves(Board, Heights, Player, Moves),
    (Moves = [] -> 
        BestMove = -1, BestScore = 0 
    ;
        Moves = [FirstMove|_],
        search_moves(Moves, Board, Heights, Player, NewDepth, Alpha, Beta, FirstMove, -10000000, BestMove, BestScore)
    ).

% 4. Standard Base Case (Depth 0)
% Calculate static board score using heuristics.
alpha_beta(Board, Heights, Player, Depth, _, _, -1, Score) :-
    Depth =< 0,
    !,
    heuristic_eval(Board, Heights, Player, Score).

% 5. Victory Check
% If the current player has won, return a high positive score.
% Prefer faster wins (+ Depth).
alpha_beta(Board, _, Player, Depth, _, _, -1, Score) :-
    ai_check_victory(Player, Board), !, 
    Score is 100000 + Depth.

% 6. Defeat Check
% If the opponent has won, return a high negative score.
alpha_beta(Board, _, Player, Depth, _, _, -1, Score) :-
    ai_change_player(Player, Opponent),
    ai_check_victory(Opponent, Board), !,
    Score is -100000 - Depth.

% 7. Draw Check
alpha_beta(Board, _, _, _, _, _, -1, 0) :-
    ai_check_draw(Board), !.

% 8. Recursive Step (The Search Loop)
% - Generate moves.
% - Sort moves (optimization).
% - Iterate through moves using search_moves.
alpha_beta(Board, Heights, Player, Depth, Alpha, Beta, BestMove, BestScore) :-
    Depth > 0,
    get_ordered_moves(Board, Heights, Player, Moves),
    (Moves = [] -> 
        BestMove = -1, BestScore = 0 
    ;
        Moves = [FirstMove|_],
        NextDepth is Depth - 1,
        search_moves(Moves, Board, Heights, Player, NextDepth, Alpha, Beta, FirstMove, -10000000, TempMove, TempScore),
        BestMove = TempMove,
        BestScore = TempScore,
        % Store result in Transposition Table
        t_record(Board, Depth, Alpha, Beta, BestScore, BestMove)
    ).

% --- Helper to iterate through moves ---
search_moves([], _, _, _, _, _, _, RecordMove, RecordScore, RecordMove, RecordScore).

search_moves([Col|RestCols], Board, Heights, Player, Depth, Alpha, Beta, 
             CurrentBestMove, CurrentBestScore, FinalMove, FinalScore) :-
    
    simulate_move(Board, Heights, Player, Col, NewBoard, NewHeights),
    ai_change_player(Player, Opponent),
    
    % Negamax recursion: Pass -Beta and -Alpha, swap players
    alpha_beta(NewBoard, NewHeights, Opponent, Depth, -Beta, -Alpha, _, OppScore),
    
    Score is -OppScore,
    
    process_result(Score, Col, RestCols, Board, Heights, Player, Depth, Alpha, Beta,
                   CurrentBestMove, CurrentBestScore, FinalMove, FinalScore).

% Case: Beta Cutoff (Pruning)
% The opponent has a move so good that this branch will never be reached.
process_result(Score, Col, _, Board, _, _, Depth, Alpha, Beta, _, _, Col, Score) :-
    Score >= Beta, !,
    t_record(Board, Depth, Alpha, Beta, Score, Col).

% Case: New Best Move Found (Raise Alpha)
process_result(Score, Col, RestCols, Board, Heights, Player, Depth, Alpha, Beta, _, _, FinalMove, FinalScore) :-
    Score > Alpha, 
    !, 
    search_moves(RestCols, Board, Heights, Player, Depth, Score, Beta, Col, Score, FinalMove, FinalScore).

% Case: Not a better move, continue searching remaining columns
process_result(_, _, RestCols, Board, Heights, Player, Depth, Alpha, Beta,
               BestMove, BestScore, FinalMove, FinalScore) :-
    search_moves(RestCols, Board, Heights, Player, Depth, Alpha, Beta, BestMove, BestScore, FinalMove, FinalScore).


% ==============================================================================
%    HEURISTIC EVALUATION
% ==============================================================================

% Calculates the static "goodness" of a board state.
% This is called when the Minimax search reaches the maximum depth.
% Formula: Score = (Score of all possible 4-cell lines) + (Center Column Bonus).
heuristic_eval(Board, Heights, Player, Score) :-
    cached_windows(AllCoords),
    ai_change_player(Player, Opponent),
    
    % 1. Evaluate all potential winning lines (horizontal, vertical, diagonal).
    %    The score is calculated relative to the Player (My Potential - Opponent Potential).
    evaluate_all_windows(AllCoords, Board, Heights, Player, Opponent, 0, LineScore),
    
    % 2. Add positional bonus for controlling the center column.
    evaluate_position_bonus(Board, Player, PosScore),
    
    Score is LineScore + PosScore.

% Strategic Heuristic: Center Control.
% In Connect 4, the center column allows for more possible winning connections 
% (vertical, horizontal, and both diagonals) than edge columns.
% We give a static bonus for every piece the AI has in the center column (Index 4).
evaluate_position_bonus(Board, Player, Score) :-
    nth1(4, Board, CenterCol),    
    count(Player, CenterCol, Count),
    Score is Count * 8.

% Recursively evaluates all precomputed windows (lines of 4 cells).
% Accumulates the score: +Score for my opportunities, -Score for opponent's threats.
evaluate_all_windows([], _, _, _, _, Acc, Acc).
evaluate_all_windows([Coords|Rest], Board, Heights, Player, Opp, CurrentScore, FinalScore) :-
    extract_values(Board, Coords, Window), 
    
    % Calculate score for Player in this specific window
    eval_window_score_optimized(Window, Coords, Heights, Player, S_My),
    
    % Calculate score for Opponent in this specific window
    eval_window_score_optimized(Window, Coords, Heights, Opp, S_Opp),
    
    % Net score for this window = My Score - Opponent Score
    NewScore is CurrentScore + S_My - S_Opp, 
    evaluate_all_windows(Rest, Board, Heights, Player, Opp, NewScore, FinalScore).

% --- Window Evaluation Logic ---

% Case 1: Mixed Window (Dead Line).
% If the window contains the opponent's pieces, it is impossible to complete 
% a line of 4 here. The value is 0.
eval_window_score_optimized(Window, _, _, Player, 0) :-
    ai_change_player(Player, Opp),
    member(Opp, Window), !.

% Case 2: 3-in-a-row (Critical Threat).
% We have 3 pieces and 1 empty spot. We must check *playability* (Gravity).
eval_window_score_optimized(Window, Coords, Heights, Player, Score) :-
    count(Player, Window, 3), 
    !, 
    (check_playability_fast(Window, Coords, Heights) ->
        % IMMEDIATE THREAT: The empty spot is reachable right now.
        % High score to encourage taking this move or blocking it.
        Score = 200 
    ;
        % FUTURE THREAT: The empty spot is higher up. 
        % Lower score because pieces must be stacked underneath first.
        Score = 15
    ).

% Case 3: 4-in-a-row (Victory).
% The game is won. Return a massive score.
eval_window_score_optimized(Window, _, _, Player, 100000) :-
    count(Player, Window, 4), !.

% Case 4: Partial Lines (1 or 2 pieces).
% Evaluates potential setup strength using base scores.
eval_window_score_optimized(Window, _, _, Player, Score) :-
    count(Player, Window, Count),
    base_score(Count, Score).

% Base scoring weights for partial lines (no immediate threat).
base_score(2, 5). % Two pieces connected: decent potential.
base_score(1, 1). % One piece: minor potential.
base_score(0, 0). % Empty window.

% Checks if the empty spot in a window is "playable" immediately.
% Connect 4 has gravity; you can only place a piece if the row matches the current column height.
check_playability_fast(Window, Coords, Heights) :-
    nth1(Index, Window, '_'),       % Find the index of the empty spot
    nth1(Index, Coords, (Col, Row)),% Get the (Col, Row) of that spot
    nth1(Col, Heights, H),          % Get the current playable height of that column
    H =:= Row,                      % Check if the empty spot is the next available slot
    !.

% ==============================================================================
%     HELPERS
% ==============================================================================

ai_check_victory(Player, Board) :-
    cached_windows(AllCoords),
    member(WindowCoords, AllCoords),
    extract_values(Board, WindowCoords, Cells),
    Cells == [Player, Player, Player, Player].

ai_check_draw(Board) :-
    \+ (nth1(_, Board, Col), member('_', Col)). % No empty cells left

ai_change_player('x', 'o').
ai_change_player('o', 'x').

% Move Ordering:
% 1. Check Transposition Table for a previously known best move.
% 2. Otherwise, check center columns first (4, 3, 5, etc.).
% This increases the chance of Alpha-Beta pruning occurring early.
get_ordered_moves(Board, Heights, _Player, OrderedMoves) :-
    ordered_valid_moves(Heights, ValidMoves), 
    
    term_hash(Board, Hash),
    (memory_table(Hash, Board, _, _, _, BestMove) ->
        (member(BestMove, ValidMoves) ->
            select(BestMove, ValidMoves, Rest),
            OrderedMoves = [BestMove | Rest]
            % write('(Memory: Try Col '), write(BestMove), write(' first) ')
        ;
            OrderedMoves = ValidMoves
        )
    ;
        OrderedMoves = ValidMoves
    ).

% Returns list of valid columns (where height <= 6), prioritized by center.
ordered_valid_moves(Heights, Moves) :-
    PreferredOrder = [4, 3, 5, 2, 6, 1, 7],
    findall(Col, (member(Col, PreferredOrder), nth1(Col, Heights, H), H =< 6), Moves).

simulate_move(Board, Heights, Player, Col, NewBoard, NewHeights) :-
    nth1(Col, Heights, H),
    H =< 6,
    set_cell_val(Board, Col, H, Player, NewBoard),
    H2 is H + 1,
    replace_val(Heights, Col, H2, NewHeights).

set_cell_val(Board, Col, Row, Val, NewBoard) :-
    nth1(Col, Board, OldCol),
    replace_val(OldCol, Row, Val, NewCol),
    replace_val(Board, Col, NewCol, NewBoard).

replace_val([_|T], 1, X, [X|T]).
replace_val([H|T], I, X, [H|R]) :- I > 1, I2 is I - 1, replace_val(T, I2, X, R).

extract_values(Board, [(C1,R1), (C2,R2), (C3,R3), (C4,R4)], [V1, V2, V3, V4]) :-
    ai_get_cell(Board, C1, R1, V1),
    ai_get_cell(Board, C2, R2, V2),
    ai_get_cell(Board, C3, R3, V3),
    ai_get_cell(Board, C4, R4, V4),
    !.

ai_get_cell(Board, C, R, Val) :-
    nth1(C, Board, ColList),
    nth1(R, ColList, Val),
    !.

% count/3 (== based) is provided by board.pl.

% --- Transposition Table Helpers ---

% Retrieves stored score if the stored depth is sufficient.
memory_lookup(Board, Depth, Alpha, Beta, Score, Move) :-
    term_hash(Board, Hash),
    memory_table(Hash, CachedBoard, MemoryDepth, Flag, MemoryScore, MemoryMove),
    Board == CachedBoard,       % Ensure exact board match (avoid hash collisions)
    MemoryDepth >= Depth,       % Ensure stored search was deep enough
    t_satisfies(Flag, MemoryScore, Alpha, Beta),
    Score = MemoryScore,
    Move = MemoryMove,
    !.

% Checks if the stored score is useful relative to current Alpha/Beta.
t_satisfies(exact, _, _, _).
t_satisfies(lower, Score, _, Beta) :- Score >= Beta.
t_satisfies(upper, Score, Alpha, _) :- Score =< Alpha.

% Records a new state in the memory table.
t_record(Board, Depth, Alpha, Beta, Score, Move) :-
    Depth > 2,
    term_hash(Board, Hash),
    determine_flag(Score, Alpha, Beta, Flag),
    (memory_table(Hash, CachedBoard, OldDepth, _, _, _) ->
        (Board == CachedBoard ->
            (Depth >= OldDepth -> 
                % Overwrite if current search is deeper (better quality)
                retract(memory_table(Hash, CachedBoard, _, _, _, _)),
                assertz(memory_table(Hash, Board, Depth, Flag, Score, Move))
            ; true)
        ;
            % Overwrite on hash collision
            retract(memory_table(Hash, _, _, _, _, _)),
            assertz(memory_table(Hash, Board, Depth, Flag, Score, Move))
        )
    ;
        % New entry
        assertz(memory_table(Hash, Board, Depth, Flag, Score, Move)) 
    ).
t_record(_, _, _, _, _, _).
% Determines if the score is exact, or an upper/lower bound based on pruning.
determine_flag(Score, _, Beta, lower) :- Score >= Beta, !. 
determine_flag(Score, Alpha, _, upper) :- Score =< Alpha, !. 
determine_flag(_, _, _, exact).