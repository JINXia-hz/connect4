%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  src/ai/nn.pl                       %%%
%%%  Neural-network-guided alpha-beta   %%%
%%%  AI. Same negamax structure as      %%%
%%%  minimax.pl, but leaf evaluation    %%%
%%%  is a value net called through      %%%
%%%  Janus (players are marks 'x'/'o'). %%%
%%%  Main predicates: nn_ia_play/4,     %%%
%%%  nn_alpha_beta/8, nn_eval/3,        %%%
%%%  nn_clean_all/0                     %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

:- use_module(library(janus)).
:- consult('../core/board.pl').
:- consult('minimax.pl').   % reuse simulate_move/6, ai_check_victory/2, not_stable/2, ...

% Make the Python bridge importable. The path is computed relative to this
% file's directory, so loading works from any current working directory.
:- (prolog_load_context(directory, SrcDir) ->
        atom_concat(SrcDir, '/../../ml', MlDir),
        catch(py_add_lib_dir(MlDir), _, true)
    ;
        true
    ).

% ==============================================================================
%    NN TRANSPOSITION TABLE
% ==============================================================================

% Separate from minimax.pl's memory_table/6: the two files are loaded
% together and their score scales are incompatible (NN evals are in
% [-1000, 1000], heuristic evals are not).
% Format: nn_memory_table(Hash, Board, Depth, Flag, Score, BestMove).
:- dynamic nn_memory_table/6.

% Clear the NN transposition table and the Python-side eval cache.
nn_clean_all :-
    retractall(nn_memory_table(_, _, _, _, _, _)),
    catch(py_call(bridge:reset_cache), _, true).

% ==============================================================================
%    AI PLAY INTERFACE
% ==============================================================================

% Fixed shallow depth: the NN eval is strong, but every leaf costs a
% Janus round-trip, so we search less deep than the heuristic AI.
% Quiescence extension (not_stable/2) still applies below.
% Depth 3 plays ~5x faster than depth 4 but loses tactically against
% deep-search opponents; 4 is the default strength/speed trade-off.
nn_depth(4).

% Minimum search depth at which the NN policy head is used for move
% ordering (one policy py_call per such node, cached bridge-side).
% Deeper (nearer-leaf) nodes keep the center-based order for speed.
% Dynamic so it is tunable at runtime (e.g. 99 disables policy ordering).
:- dynamic nn_policy_order_min_depth/1.
nn_policy_order_min_depth(2).

set_nn_policy_order_min_depth(D) :-
    retractall(nn_policy_order_min_depth(_)),
    assertz(nn_policy_order_min_depth(D)).

% Main entry point for the NN AI to make a move.
nn_ia_play(Board, Heights, Player, BestCol) :-
    nn_depth(Depth),
    write('>>> NN Move: '), write(Player),
    write(' | Depth: '), write(Depth), nl,
    (nn_alpha_beta(Board, Heights, Player, Depth, -9999999, 9999999, BestCol, Score) ->
        write('>>> Decision: Col '), write(BestCol), write(' (Score '), write(Score), write(')'), nl
    ;
        write('>>> NN Fail! Fallback random.'), nl, fail
    ).

% ==============================================================================
%    NEURAL EVALUATION
% ==============================================================================

% Static evaluation by the value net, scaled to integers in [-1000, 1000]
% (far below the +/-100000 terminal scores, so the integer alpha-beta
% arithmetic of the heuristic AI still works unchanged).
nn_eval(Board, Player, Score) :-
    catch(py_call(bridge:eval(Board, Player), V), _, fail),
    !,
    Score is round(V * 1000).

% Fallback when Python/Janus is unavailable: reuse the hand-written
% heuristic so the AI never crashes mid-game. Heights are derived from
% the board (height = 7 - number of empty cells in the column).
nn_eval(Board, Player, Score) :-
    board_heights(Board, Heights),
    heuristic_eval(Board, Heights, Player, Score).

board_heights(Board, Heights) :-
    maplist(column_height, Board, Heights).

column_height(Col, H) :-
    empty_mark(E),
    count(E, Col, Empty),
    H is 7 - Empty.

% ==============================================================================
%    MINIMAX WITH ALPHA-BETA PRUNING (NN eval)
% ==============================================================================

% 1. Transposition Table Lookup
nn_alpha_beta(Board, _, _, Depth, Alpha, Beta, BestMove, BestScore) :-
    nn_memory_lookup(Board, Depth, Alpha, Beta, CachedScore, CachedMove),
    !,
    BestScore = CachedScore,
    BestMove = CachedMove.

% 2. Hard Depth Limit (extended search stops here)
nn_alpha_beta(Board, _, Player, Depth, _, _, -1, Score) :-
    Depth =< -2,
    !,
    nn_eval(Board, Player, Score).

% 3. Search Extension (Quiescence-like)
% If Depth is 0 but the board is "not stable" (immediate threat exists),
% extend the search by 1 level.
nn_alpha_beta(Board, Heights, Player, Depth, Alpha, Beta, BestMove, BestScore) :-
    Depth =< 0,
    not_stable(Board, Heights),
    !,
    NewDepth is Depth - 1,
    nn_get_ordered_moves(Board, Heights, Player, NewDepth, Moves),
    (Moves = [] ->
        BestMove = -1, BestScore = 0
    ;
        Moves = [FirstMove|_],
        nn_search_moves(Moves, Board, Heights, Player, NewDepth, Alpha, Beta, FirstMove, -10000000, BestMove, BestScore)
    ).

% 4. Standard Base Case (Depth 0): static NN evaluation
nn_alpha_beta(Board, _, Player, Depth, _, _, -1, Score) :-
    Depth =< 0,
    !,
    nn_eval(Board, Player, Score).

% 5. Victory Check (prefer faster wins: + Depth)
nn_alpha_beta(Board, _, Player, Depth, _, _, -1, Score) :-
    ai_check_victory(Player, Board), !,
    Score is 100000 + Depth.

% 6. Defeat Check
nn_alpha_beta(Board, _, Player, Depth, _, _, -1, Score) :-
    ai_change_player(Player, Opponent),
    ai_check_victory(Opponent, Board), !,
    Score is -100000 - Depth.

% 7. Draw Check
nn_alpha_beta(Board, _, _, _, _, _, -1, 0) :-
    ai_check_draw(Board), !.

% 8. Recursive Step (The Search Loop)
nn_alpha_beta(Board, Heights, Player, Depth, Alpha, Beta, BestMove, BestScore) :-
    Depth > 0,
    nn_get_ordered_moves(Board, Heights, Player, Depth, Moves),
    (Moves = [] ->
        BestMove = -1, BestScore = 0
    ;
        Moves = [FirstMove|_],
        NextDepth is Depth - 1,
        nn_search_moves(Moves, Board, Heights, Player, NextDepth, Alpha, Beta, FirstMove, -10000000, TempMove, TempScore),
        BestMove = TempMove,
        BestScore = TempScore,
        % Store result in the NN Transposition Table
        nn_t_record(Board, Depth, Alpha, Beta, BestScore, BestMove)
    ).

% --- Helper to iterate through moves ---
nn_search_moves([], _, _, _, _, _, _, RecordMove, RecordScore, RecordMove, RecordScore).

nn_search_moves([Col|RestCols], Board, Heights, Player, Depth, Alpha, Beta,
                CurrentBestMove, CurrentBestScore, FinalMove, FinalScore) :-

    simulate_move(Board, Heights, Player, Col, NewBoard, NewHeights),
    ai_change_player(Player, Opponent),

    % Negamax recursion: pass -Beta and -Alpha, swap players
    nn_alpha_beta(NewBoard, NewHeights, Opponent, Depth, -Beta, -Alpha, _, OppScore),

    Score is -OppScore,

    nn_process_result(Score, Col, RestCols, Board, Heights, Player, Depth, Alpha, Beta,
                      CurrentBestMove, CurrentBestScore, FinalMove, FinalScore).

% Case: Beta Cutoff (Pruning)
nn_process_result(Score, Col, _, Board, _, _, Depth, Alpha, Beta, _, _, Col, Score) :-
    Score >= Beta, !,
    nn_t_record(Board, Depth, Alpha, Beta, Score, Col).

% Case: New Best Move Found (Raise Alpha)
nn_process_result(Score, Col, RestCols, Board, Heights, Player, Depth, Alpha, Beta, _, _, FinalMove, FinalScore) :-
    Score > Alpha,
    !,
    nn_search_moves(RestCols, Board, Heights, Player, Depth, Score, Beta, Col, Score, FinalMove, FinalScore).

% Case: Not a better move, continue searching remaining columns
nn_process_result(_, _, RestCols, Board, Heights, Player, Depth, Alpha, Beta,
                  BestMove, BestScore, FinalMove, FinalScore) :-
    nn_search_moves(RestCols, Board, Heights, Player, Depth, Alpha, Beta, BestMove, BestScore, FinalMove, FinalScore).

% ==============================================================================
%     HELPERS
% ==============================================================================

% Move Ordering: NN-TT best move first; the remaining valid moves are
% ordered by the NN policy head (probability descending) at near-root
% nodes (Depth >= nn_policy_order_min_depth), or center-first deeper.
% Improves pruning. Falls back to center order if Python/Janus fails.
nn_get_ordered_moves(Board, Heights, Player, Depth, OrderedMoves) :-
    ordered_valid_moves(Heights, ValidMoves),

    term_hash(Board, Hash),
    (nn_memory_table(Hash, Board, _, _, _, BestMove) ->
        (member(BestMove, ValidMoves) ->
            select(BestMove, ValidMoves, Rest),
            TtOrdered = [BestMove | Rest]
        ;
            TtOrdered = ValidMoves
        )
    ;
        TtOrdered = ValidMoves
    ),

    (   nn_policy_order_min_depth(MinDepth),
        Depth >= MinDepth,
        catch(py_call(bridge:policy(Board, Player), Probs), _, fail)
    ->  order_by_policy(TtOrdered, Probs, OrderedMoves)
    ;   OrderedMoves = TtOrdered
    ).

% Order columns by policy probability descending (keysort is stable,
% so ties keep the incoming order). Probs covers all 7 columns;
% only the valid ones are ordered here (full columns already masked).
order_by_policy(Moves, Probs, Ordered) :-
    findall(P-C, (member(C, Moves), nth1(C, Probs, P)), Pairs),
    keysort(Pairs, Asc),
    reverse(Asc, Desc),
    findall(C, member(_-C, Desc), Ordered).

% --- Transposition Table Helpers ---
% t_satisfies/4 and determine_flag/4 from minimax.pl are reused as-is:
% they are pure flag/score logic, independent of the evaluation scale.

% Retrieves stored score if the stored depth is sufficient.
nn_memory_lookup(Board, Depth, Alpha, Beta, Score, Move) :-
    term_hash(Board, Hash),
    nn_memory_table(Hash, CachedBoard, MemoryDepth, Flag, MemoryScore, MemoryMove),
    Board == CachedBoard,       % Ensure exact board match (avoid hash collisions)
    MemoryDepth >= Depth,       % Ensure stored search was deep enough
    t_satisfies(Flag, MemoryScore, Alpha, Beta),
    Score = MemoryScore,
    Move = MemoryMove,
    !.

% Records a new state in the NN memory table.
nn_t_record(Board, Depth, Alpha, Beta, Score, Move) :-
    Depth > 2,
    term_hash(Board, Hash),
    determine_flag(Score, Alpha, Beta, Flag),
    (nn_memory_table(Hash, CachedBoard, OldDepth, _, _, _) ->
        (Board == CachedBoard ->
            (Depth >= OldDepth ->
                % Overwrite if current search is deeper (better quality)
                retract(nn_memory_table(Hash, CachedBoard, _, _, _, _)),
                assertz(nn_memory_table(Hash, Board, Depth, Flag, Score, Move))
            ; true)
        ;
            % Overwrite on hash collision
            retract(nn_memory_table(Hash, _, _, _, _, _)),
            assertz(nn_memory_table(Hash, Board, Depth, Flag, Score, Move))
        )
    ;
        % New entry
        assertz(nn_memory_table(Hash, Board, Depth, Flag, Score, Move))
    ).
nn_t_record(_, _, _, _, _, _).
