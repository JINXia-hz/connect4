%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  src/ai/mcts.pl                     %%%
%%%  Monte Carlo Tree Search AI with    %%%
%%%  dynamic predicates.                %%%
%%%  Main predicates: mcts_play/4,      %%%
%%%  set_mcts_iterations/1              %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

:- use_module(library(lists)).
:- use_module(library(random)).
:- consult('../core/board.pl').

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%    DYNAMIC TREE STORAGE            %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% These predicates are modified dynamically during the search
:- dynamic mcts_node/8.
% mcts_node(NodeID, Board, Heights, Player, ParentID, Visits, Wins, UntriedMoves)
% A node holds: its ID, the game state, the player, its parent,
% its stats (visits/wins), and the moves not tried yet.

:- dynamic mcts_child/2.
% mcts_child(ParentID, ChildID): links a parent to its children in the tree

:- dynamic next_node_id/1.
% Generates unique node IDs

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%    MCTS CONFIGURATION              %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Number of simulations per move (dynamically settable, e.g. by the
% self-play generator which uses MCTS as a teacher with fewer iterations)
:- dynamic mcts_iterations/1.
mcts_iterations(3000).

% Set the number of MCTS simulations per move
set_mcts_iterations(N) :-
    retractall(mcts_iterations(_)),
    assertz(mcts_iterations(N)).

exploration_constant(1.414).  % Constant C in the UCB1 formula (sqrt(2))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%    MAIN INTERFACE                  %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mcts_play(Board, Heights, Player, BestCol) :-
    write('>>> IA MCTS réfléchit...'), nl,

    % Clear the tree from the previous game to start fresh
    retractall(mcts_node(_, _, _, _, _, _, _, _)),
    retractall(mcts_child(_, _)),
    retractall(next_node_id(_)),
    asserta(next_node_id(1)),

    % Create the root node representing the current game position
    get_valid_moves(Heights, ValidMoves),
    create_node(Board, Heights, Player, root, ValidMoves, RootID),

    % Run the MCTS search with the configured number of iterations
    mcts_iterations(NumIter),
    mcts_search(RootID, NumIter),

    % After all simulations, pick the best move
    select_best_move(RootID, BestCol),
    write('>>> IA MCTS joue la colonne '), write(BestCol), nl.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%    NODE MANAGEMENT                 %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

create_node(Board, Heights, Player, ParentID, UntriedMoves, NodeID) :-
    % Grab the next available ID and bump the counter
    next_node_id(NodeID),
    NewID is NodeID + 1,
    retract(next_node_id(_)),
    asserta(next_node_id(NewID)),

    % The node starts with 0 visits and 0 wins
    assertz(mcts_node(NodeID, Board, Heights, Player, ParentID, 0, 0, UntriedMoves)),

    % Unless it is the root, register it as a child of its parent
    (ParentID \= root ->
        assertz(mcts_child(ParentID, NodeID))
    ;
        true
    ).

% All child IDs of a given node
get_children(NodeID, Children) :-
    findall(ChildID, mcts_child(NodeID, ChildID), Children).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%    MOVE UTILITIES                  %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% All columns (1 to 7) whose height is <= 6 (not full)
get_valid_moves(Heights, Moves) :-
    findall(Col,
            (between(1, 7, Col),
             nth1(Col, Heights, H),
             H =< 6),
            Moves).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%    GAME STATE UTILITIES            %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% A state is terminal if someone won or it is a draw
is_terminal(Board, Heights) :-
    (check_victory('x', Board) ;
     check_victory('o', Board) ;
     is_draw(Board, Heights)).

% Draw when every column is full (height > 6)
is_draw(_, Heights) :-
    \+ (member(H, Heights), H =< 6).

get_winner(Board, Winner) :-
    (check_victory('x', Board) -> Winner = 'x'
    ; check_victory('o', Board) -> Winner = 'o'
    ; Winner = draw).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%    SIMULATION (ROLLOUT)            %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Terminal state: evaluate the result
rollout(Board, Heights, _, Result) :-
    is_terminal(Board, Heights), !,
    get_winner(Board, Winner),
    winner_to_score(Winner, Result).

% Otherwise play a random move and keep going
rollout(Board, Heights, Player, Result) :-
    get_valid_moves(Heights, ValidMoves),
    ValidMoves \= [], !,
    random_member(Col, ValidMoves),
    play_move(Board, Heights, Player, Col, NewBoard, NewHeights),
    next_player(Player, NextPlayer),
    rollout(NewBoard, NewHeights, NextPlayer, Result).

% Fallback when no move is possible
rollout(Board, _, _, Result) :-
    get_winner(Board, Winner),
    winner_to_score(Winner, Result).

% Convert the winner to a numeric score
winner_to_score('x', 1).    % X wins = +1 (player 1)
winner_to_score('o', -1).   % O wins = -1 (player 2)
winner_to_score(draw, 0).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%    UCB1 COMPUTATION                %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

uct_value(NodeID, ParentVisits, UCTValue) :-
    mcts_node(NodeID, _, _, _, _, Visits, Wins, _),
    Visits > 0, !,
    exploration_constant(C),

    % Wins/Visits is the win rate of the node's player, but it is
    % evaluated from the parent (opponent), so it is inverted
    Exploitation is 1 - (Wins / Visits),

    % Exploration term: encourages visiting rarely explored nodes
    Exploration is C * sqrt(log(ParentVisits) / Visits),
    UCTValue is Exploitation + Exploration.

% Never-visited nodes get a huge value to force exploration
uct_value(_, _, 999999).

% Child with the highest UCB1 value
best_child_uct(NodeID, BestChildID) :-
    mcts_node(NodeID, _, _, _, _, ParentVisits, _, _),
    get_children(NodeID, Children),
    Children \= [],
    best_child_uct_helper(Children, ParentVisits, nil, -999999, BestChildID).

best_child_uct_helper([], _, BestID, _, BestID).
best_child_uct_helper([ChildID|Rest], ParentVisits, CurrentBest, CurrentUCT, BestID) :-
    uct_value(ChildID, ParentVisits, UCT),
    (UCT > CurrentUCT ->
        best_child_uct_helper(Rest, ParentVisits, ChildID, UCT, BestID)
    ;
        best_child_uct_helper(Rest, ParentVisits, CurrentBest, CurrentUCT, BestID)
    ).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%    PHASE 1: SELECTION              %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Terminal node: stop here
select_node(NodeID, SelectedID) :-
    mcts_node(NodeID, Board, Heights, _, _, _, _, _),
    is_terminal(Board, Heights), !,
    SelectedID = NodeID.

% Node with untried moves: stop here for expansion
select_node(NodeID, SelectedID) :-
    mcts_node(NodeID, _, _, _, _, _, _, UntriedMoves),
    UntriedMoves \= [], !,
    SelectedID = NodeID.

% Node without children: stop here
select_node(NodeID, SelectedID) :-
    get_children(NodeID, []), !,
    SelectedID = NodeID.

% Otherwise descend towards the child with the best UCB1
select_node(NodeID, SelectedID) :-
    best_child_uct(NodeID, BestChildID),
    select_node(BestChildID, SelectedID).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%    PHASE 2: EXPANSION              %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

expand_node(NodeID, NewChildID) :-
    mcts_node(NodeID, Board, Heights, Player, _, Visits, Wins, UntriedMoves),
    UntriedMoves \= [],

    % Pick a RANDOM untried move (avoids bias towards the left)
    random_member(Col, UntriedMoves),
    select(Col, UntriedMoves, RestUntried),

    % Update the parent's untried move list
    retract(mcts_node(NodeID, Board, Heights, Player, ParentID, Visits, Wins, _)),
    assertz(mcts_node(NodeID, Board, Heights, Player, ParentID, Visits, Wins, RestUntried)),

    % Play the chosen move
    play_move(Board, Heights, Player, Col, NewBoard, NewHeights),
    next_player(Player, NextPlayer),

    % Create a new child node for this move
    get_valid_moves(NewHeights, ValidMoves),
    create_node(NewBoard, NewHeights, NextPlayer, NodeID, ValidMoves, NewChildID).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%    PHASE 4: BACKPROPAGATION        %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

backpropagate(root, _) :- !.

backpropagate(NodeID, Result) :-
    mcts_node(NodeID, Board, Heights, Player, ParentID, Visits, Wins, Untried),

    NewVisits is Visits + 1,

    % Update wins according to the result and the node player
    (Player =:= 1 ->
        (Result =:= 1 -> NewWins is Wins + 1
        ; Result =:= 0 -> NewWins is Wins + 0.5
        ; NewWins = Wins)
    ;
        (Result =:= -1 -> NewWins is Wins + 1
        ; Result =:= 0 -> NewWins is Wins + 0.5
        ; NewWins = Wins)
    ),

    % Store the updated stats
    retract(mcts_node(NodeID, _, _, _, _, _, _, _)),
    assertz(mcts_node(NodeID, Board, Heights, Player, ParentID, NewVisits, NewWins, Untried)),

    % Keep propagating up to the parent
    backpropagate(ParentID, Result).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%    MAIN MCTS LOOP                  %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mcts_search(_, 0) :- !.

mcts_search(RootID, NumIter) :-
    NumIter > 0,
    mcts_iteration(RootID),
    NextIter is NumIter - 1,
    mcts_search(RootID, NextIter).

% One iteration = 4 phases: Selection, Expansion, Simulation, Backpropagation
mcts_iteration(RootID) :-
    % Phase 1: Selection - walk down the tree
    select_node(RootID, SelectedID),

    mcts_node(SelectedID, Board, Heights, Player, _, _, _, _),

    (is_terminal(Board, Heights) ->
        % Terminal node: evaluate and backpropagate
        get_winner(Board, Winner),
        winner_to_score(Winner, Result),
        backpropagate(SelectedID, Result)
    ;
        mcts_node(SelectedID, _, _, _, _, _, _, UntriedMoves),
        (UntriedMoves \= [] ->
            % Phase 2: Expansion - create a new node
            expand_node(SelectedID, NewChildID),
            mcts_node(NewChildID, ChildBoard, ChildHeights, ChildPlayer, _, _, _, _),
            % Phase 3: Simulation - random playout until the end
            rollout(ChildBoard, ChildHeights, ChildPlayer, Result),
            % Phase 4: Backpropagation - propagate the result up
            backpropagate(NewChildID, Result)
        ;
            % Fully expanded: simulate directly from the selected node
            rollout(Board, Heights, Player, Result),
            backpropagate(SelectedID, Result)
        )
    ).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%    BEST MOVE SELECTION             %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% After all simulations, pick the most visited child
select_best_move(RootID, BestCol) :-
    get_children(RootID, Children),
    Children \= [],
    most_visited_child(Children, BestChildID),

    % Recover the move played between the root and this child
    mcts_node(RootID, ParentBoard, _, _, _, _, _, _),
    mcts_node(BestChildID, ChildBoard, _, _, _, Visits, Wins, _),
    find_played_column(ParentBoard, ChildBoard, BestCol),

    % Display statistics for debugging
    (Visits > 0 -> WinRate is Wins / Visits ; WinRate is 0),
    write('>>> Colonne sélectionnée : '), write(BestCol),
    write(' (Visits: '), write(Visits),
    write(', WinRate: '), format('~2f', [WinRate]), write(')'), nl.

% Fallback when there are no children: play the first valid move
select_best_move(RootID, BestCol) :-
    mcts_node(RootID, _, Heights, _, _, _, _, _),
    get_valid_moves(Heights, [BestCol|_]).

most_visited_child([ChildID], ChildID) :- !.

most_visited_child([Child1|Rest], MostVisited) :-
    mcts_node(Child1, _, _, _, _, V1, _, _),
    most_visited_child(Rest, TempBest),
    mcts_node(TempBest, _, _, _, _, V2, _, _),
    (V1 > V2 ->
        MostVisited = Child1
    ;
        MostVisited = TempBest
    ).

% Find which column changed between parent and child boards
find_played_column(ParentBoard, ChildBoard, Col) :-
    between(1, 7, Col),
    nth1(Col, ParentBoard, ParentCol),
    nth1(Col, ChildBoard, ChildCol),
    ParentCol \= ChildCol, !.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%    DEBUG                           %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Print the number of nodes in the tree
print_tree_stats :-
    findall(1, mcts_node(_, _, _, _, _, _, _, _), Nodes),
    length(Nodes, NumNodes),
    write('Nombre de nœuds dans l\'arbre: '), write(NumNodes), nl.
