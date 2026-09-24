%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  src/game/puissance4.pl             %%%
%%%  Connect Four - complete game:      %%%
%%%  menus, game loop, board display.   %%%
%%%  Main predicates: jouer/0,          %%%
%%%  setup_game/2, start_game/0,        %%%
%%%  start_game/1, get_move_by_type/5   %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

:- consult('../core/board.pl').
:- consult('../ai/random.pl').
:- consult('../core/victory.pl').
:- consult('../ai/minimax.pl').        % Minimax with transposition table
:- consult('../ai/nn.pl').             % Neural-network-guided alpha-beta (via Janus)
:- consult('../ai/alphabeta.pl').      % Pedagogical alpha-beta minimax
:- consult('../ai/mcts.pl').

% These predicates are modified dynamically to store the game state
:- dynamic board/1.
:- dynamic heights/1.
:- dynamic player_type/2.
:- dynamic ai_delay/1.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%     GAME INITIALIZATION            %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initialize the board and the column heights
initialize(Board, Heights) :-
    empty_mark(E),
    make_list(6, E, EmptyCol),
    make_list(7, EmptyCol, Board),
    make_list(7, 1, Heights).

% Create a list of N identical items
make_list(0, _, []) :- !.
make_list(N, Item, [Item|T]) :-
    N > 0,
    N2 is N - 1,
    make_list(N2, Item, T).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%     BOARD DISPLAY                  %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

output_board(Board) :-
    num_rows(MaxRow),
    nl,
    write('  1   2   3   4   5   6   7  '), nl,
    write('-----------------------------'), nl,
    print_rows(Board, MaxRow),
    write('-----------------------------'), nl.

print_rows(_, 0) :- !.
print_rows(Board, RowNum) :-
    write('|'),
    print_row_cells(Board, RowNum),
    nl,
    NextRow is RowNum - 1,
    print_rows(Board, NextRow).

print_row_cells([], _) :- !.
print_row_cells([Col|RestCols], RowNum) :-
    get_element(Col, RowNum, Val),
    output_square(Val),
    write('|'),
    print_row_cells(RestCols, RowNum).

output_square(Val) :-
    empty_mark(E),
    Val == E, !,
    write('   ').
output_square(Val) :-
    write(' '), write(Val), write(' ').

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%     GAME OVER                      %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% The game is over when someone won or the board is full
game_over(Board) :-
	(check_victory('x', Board) ; check_victory('o', Board)), !.

game_over(Board) :-
	check_draw(Board).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%     GAME MENU                      %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Display the menu and handle the user's choice
jouer :-
    nl,
    write('================================='), nl,
    write('===   PUISSANCE 4 - MENU      ==='), nl,
    write('================================='), nl, nl,
    write('1. Jouer Humain vs Humain'), nl,
    write('2. Jouer Humain vs IA'), nl,
    write('3. Jouer IA vs IA'), nl,
    write('4. Quitter'), nl, nl,
    write('Votre choix (1-4): '),
    read(Choice),
    handle_menu_choice(Choice).

% Main menu choice handling
% Option 1: Human vs Human
handle_menu_choice(1) :-
	!,
	setup_game(human, human),
    nl, write('=== Début de la partie Humain vs Humain ==='), nl,
	start_game.

% Option 2: Human vs AI - open the AI submenu
handle_menu_choice(2) :-
	!, nl, write('=== Choix de l\'IA ==='), nl,
    write('1. Jouer Humain vs IA (Minimax - Tres Fort)'), nl,
    write('2. Jouer Humain vs IA (Random - Facile)'), nl,
    write('3. Jouer Humain vs IA (Random+ - Moyen : Attaque)'), nl,
    write('4. Jouer Humain vs IA (Random++ - Moyen+ : Attaque/Defense)'), nl,
    write('5. Jouer Humain vs IA (MCTS - Fort : Monte Carlo)'), nl,
    write('6. Jouer Humain vs IA (Alpha-Beta - Fort : Pedagogique)'), nl,
    write('7. Jouer Humain vs IA (Neural Network - NN)'), nl, nl,
    write('0. Retour au menu principal'), nl,
    write('Votre choix (0-7): '),
    read(Choice),
    handle_ia_choice(Choice).

% Option 3: AI vs AI
handle_menu_choice(3) :-
    !,
    nl, write('=== Choix des IA ==='), nl,
    write('Joueur 1 (x) - Choisissez l\'IA :'), nl,
    write('1. Minimax (Tres Fort)'), nl,
    write('2. MCTS (Fort)'), nl,
    write('3. Random++ (Moyen+)'), nl,
    write('4. Random+ (Moyen)'), nl,
    write('5. Random (Facile)'), nl,
    write('6. Alpha-Beta (Fort - Pedagogique)'), nl,
    write('7. Neural Network (NN)'), nl,
    write('Votre choix (1-7): '),
    read(Choice1),

    nl, write('Joueur 2 (o) - Choisissez l\'IA :'), nl,
    write('1. Minimax (Tres Fort)'), nl,
    write('2. MCTS (Fort)'), nl,
    write('3. Random++ (Moyen+)'), nl,
    write('4. Random+ (Moyen)'), nl,
    write('5. Random (Facile)'), nl,
    write('6. Alpha-Beta (Fort - Pedagogique)'), nl,
    write('7. Neural Network (NN)'), nl,
    write('Votre choix (1-7): '),
    read(Choice2),

    ai_type(Choice1, AI1),
    ai_type(Choice2, AI2),

    nl, write('Voulez-vous un délai entre les coups? (secondes, 0=non): '),
    read(Delay),
    retractall(ai_delay(_)),
    asserta(ai_delay(Delay)),

    setup_game(AI1, AI2),
    nl, write('=== Début de la partie IA vs IA ==='), nl,
    write('Joueur 1 (x): '), describe_ai(AI1), nl,
    write('Joueur 2 (o): '), describe_ai(AI2), nl, nl,
    start_game.

% Option 4: Quit
handle_menu_choice(4) :-
    !,
    nl, write('Au revoir !'), nl.

% Choice -> AI type conversion
ai_type(1, computer).
ai_type(2, computer_mcts).
ai_type(3, computer_random_plus).
ai_type(4, computer_evolue).
ai_type(5, computer_random).
ai_type(6, computer_alphabeta).
ai_type(7, computer_nn).

% AI descriptions
describe_ai(computer) :- write('Minimax (Alpha-Beta + TT)').
describe_ai(computer_mcts) :- write('MCTS (Monte Carlo Tree Search)').
describe_ai(computer_random_plus) :- write('Random++ (Attaque/Defense)').
describe_ai(computer_evolue) :- write('Random+ (Attaque)').
describe_ai(computer_random) :- write('Random').
describe_ai(computer_alphabeta) :- write('Alpha-Beta (Pedagogique)').
describe_ai(computer_nn) :- write('Neural Network (Alpha-Beta + NN eval)').

%%%% AI options %%%%
% Option 1: Human vs Minimax AI
handle_ia_choice(1) :-
	!,
	nl, write('Voulez-vous jouer en premier? (1=Oui, 2=Non): '),
	read(Order),
	(   Order == 1
	->  setup_game(human, computer)
	;   setup_game(computer, human)
	),
    nl, write('=== Début de la partie Humain vs IA Minimax ==='), nl,
	start_game.

% Option 2: Human vs Random
handle_ia_choice(2) :-
    !,
    nl, write('Voulez-vous jouer en premier? (1=Oui, 2=Non): '),
    read(Order),
    (   Order == 1
    ->  setup_game(human, computer_random)
    ;   setup_game(computer_random, human)
    ),
    nl, write('=== Début de la partie Humain vs IA Random ==='), nl,
    start_game.

% Option 3: Human vs Random+
handle_ia_choice(3) :-
    !,
    nl, write('Voulez-vous jouer en premier? (1=Oui, 2=Non): '),
    read(Order),
    (   Order == 1
    ->  setup_game(human, computer_evolue)
    ;   setup_game(computer_evolue, human)
    ),
    nl, write('=== Début de la partie Humain vs IA Évoluée ==='), nl,
    start_game.

% Option 4: Human vs Random++ (Attack + Defense)
handle_ia_choice(4) :-
    !,
    nl, write('Voulez-vous jouer en premier? (1=Oui, 2=Non): '),
    read(Order),
    (   Order == 1
    ->  setup_game(human, computer_random_plus)
    ;   setup_game(computer_random_plus, human)
    ),
    nl, write('=== Début de la partie Humain vs IA Random++ ==='), nl,
    start_game.

% Option 5: Human vs MCTS AI
handle_ia_choice(5) :-
    !,
    nl, write('Voulez-vous jouer en premier? (1=Oui, 2=Non): '),
    read(Order),
    (   Order == 1
    ->  setup_game(human, computer_mcts)
    ;   setup_game(computer_mcts, human)
    ),
    nl, write('=== Début de la partie Humain vs IA MCTS ==='), nl,
    start_game.

% Option 6: Human vs Alpha-Beta AI (pedagogical)
handle_ia_choice(6) :-
    !,
    nl, write('Voulez-vous jouer en premier? (1=Oui, 2=Non): '),
    read(Order),
    (   Order == 1
    ->  setup_game(human, computer_alphabeta)
    ;   setup_game(computer_alphabeta, human)
    ),
    nl, write('=== Début de la partie Humain vs IA Alpha-Beta ==='), nl,
    start_game.

% Option 7: Human vs Neural Network AI
handle_ia_choice(7) :-
    !,
    nl, write('Voulez-vous jouer en premier? (1=Oui, 2=Non): '),
    read(Order),
    (   Order == 1
    ->  setup_game(human, computer_nn)
    ;   setup_game(computer_nn, human)
    ),
    nl, write('=== Début de la partie Humain vs IA NN ==='), nl,
    start_game.

% Option 0: Back to the main menu
handle_ia_choice(0) :-
    !,
    jouer.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%     GAME LOOP                      %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Register the player types
setup_game(Type1, Type2) :-
    % Clean up the player types from the previous game
	retractall(player_type(_, _)),
    % Record the new player types
	asserta(player_type(1, Type1)),
	asserta(player_type(2, Type2)).

% Start the game
start_game :-
	initialize(Board, Heights),
	retractall(board(_)),
	retractall(heights(_)),
    (current_predicate(clean_all/0) -> clean_all ; true),
    (current_predicate(nn_clean_all/0) -> nn_clean_all ; true),
	asserta(board(Board)),
	asserta(heights(Heights)),
	nl, nl, write('=== PUISSANCE 4 ==='), nl,
	output_board(Board),
	play(1),
	nl, nl, write('Merci d\'avoir joué !'), nl, nl,
	jouer.

start_game(R) :-
	initialize(Board, Heights),
	retractall(board(_)),
	retractall(heights(_)),
    (current_predicate(clean_all/0) -> clean_all ; true),
    (current_predicate(nn_clean_all/0) -> nn_clean_all ; true),
	asserta(board(Board)),
	asserta(heights(Heights)),
	play(1, R).


% MAIN GAME LOOP
play(Player) :-
	board(Board),
	heights(Heights),
	\+ game_over(Board),
	make_move(Player, Board, Heights),
	!,
	next_player(Player, NextPlayer),
	play(NextPlayer).

play(_) :-
	board(Board),
	nl, nl, write('=== FIN DE LA PARTIE ==='), nl,
	output_board(Board),
	output_winner(Board).

play(Player, R) :-
	board(Board),
	heights(Heights),
	\+ game_over(Board),
	make_move(Player, Board, Heights),
	!,
	next_player(Player, NextPlayer),
	play(NextPlayer, R).

play(_, R) :-
	board(Board),
	output_winner(Board, R).

make_move(Player, Board, Heights) :-
	nl, nl, write('--- Joueur '), write(Player), write(' ---'), nl,
	player_mark(Player, Mark),
	player_type(Player, Type),
	write('C\'est à '), write(Mark), write(' de jouer ('), write(Type), write(').'), nl,
	get_move_by_type(Type, Player, Board, Heights, Col),
	play_move(Board, Heights, Player, Col, NewBoard, NewHeights),
	retract(board(_)),
	retract(heights(_)),
	asserta(board(NewBoard)),
	asserta(heights(NewHeights)),
	output_board(NewBoard),
	% Delay between moves for AI vs AI games
	apply_ai_delay(Type).

% Apply the delay if the player is an AI (for AI vs AI)
apply_ai_delay(human) :- !.  % No delay for humans
apply_ai_delay(_) :-
    (ai_delay(Delay), Delay > 0 ->
        sleep(Delay)
    ;
        true
    ).

% Get the move according to the player type
get_move_by_type(human, Player, Board, Heights, Col) :-
    write('Choisissez une colonne (1-7): '),
	read(Input),
	validate_move(Input, Heights, Col, Player, Board).

% --- random ---
get_move_by_type(computer_random, _Player, _Board, Heights, Col) :-
    random_play(Heights, Col),
    write('L\'IA joue la colonne '), write(Col), nl.

% --- random+ ---
get_move_by_type(computer_evolue, Player, Board, Heights, Col) :-
    smart_random_play(Board, Heights, Player, Col),
    write('L\'IA joue la colonne '), write(Col), nl.

% --- random++ ---
get_move_by_type(computer_random_plus, Player, Board, Heights, Col) :-
    write('L\'IA (Random++) reflechit...'), nl,
    sleep(1),
    defensive_random_play(Board, Heights, Player, Col),
    write('L\'IA joue la colonne '), write(Col), nl.

% --- Minimax AI (transposition table, works with marks 'x'/'o') ---
get_move_by_type(computer, Player, Board, Heights, Col) :-
	write('L\'IA Minimax réfléchit...'), nl,

    player_mark(Player, Mark),

    catch(
        ia_play(Board, Heights, Mark, Col),
        error(resource_error(stack), _),
        (
            write('Stack overflow IA -> random fallback'), nl,
            random_play(Heights, Col)
        )
    ),

	write('L\'IA joue la colonne '), write(Col), nl.

% --- Neural Network AI (alpha-beta + NN eval via Janus) ---
get_move_by_type(computer_nn, Player, Board, Heights, Col) :-
	write('L\'IA NN réfléchit...'), nl,

    player_mark(Player, Mark),

    catch(
        nn_ia_play(Board, Heights, Mark, Col),
        error(resource_error(stack), _),
        (
            write('Stack overflow IA -> random fallback'), nl,
            random_play(Heights, Col)
        )
    ),

	write('L\'IA joue la colonne '), write(Col), nl.

% --- MCTS AI ---
get_move_by_type(computer_mcts, Player, Board, Heights, Col) :-
    mcts_play(Board, Heights, Player, Col).

% --- Alpha-Beta AI (pedagogical) ---
get_move_by_type(computer_alphabeta, Player, Board, Heights, Col) :-
    write('L\'IA Alpha-Beta réfléchit...'), nl,
    catch(
        alpha_beta(Board, Heights, 8, Player, simple_eval, -999999, 999999, Col, Score),
        error(resource_error(stack), _),
        (
            write('Stack overflow -> lower depth fallback'), nl,
            catch(
                alpha_beta(Board, Heights, 7, Player, simple_eval, -999999, 999999, Col, Score),
                error(resource_error(stack), _),
                (
                    write('Stack overflow -> lower depth fallback'), nl,
                    alpha_beta(Board, Heights, 5, Player, simple_eval, -999999, 999999, Col, Score)
                )
            )
        )
    ),
    write('L\'IA joue la colonne '), write(Col),
    write(' (Score: '), write(Score), write(')'), nl.

% Move validation
validate_move(Input, Heights, Col, _, _) :-
	integer(Input),
	Input >= 1,
	Input =< 7,
	nth1(Input, Heights, H),
	num_rows(MaxRow),
	H =< MaxRow,
	!,
	Col = Input.

validate_move(_, Heights, Col, Player, Board) :-
	write('Colonne invalide ou pleine. Réessayez.'), nl,
	get_move_by_type(human, Player, Board, Heights, Col).

% Display the winner
output_winner(Board) :-
	check_victory('x', Board),
	write('Le joueur x a gagné !'), nl, !.

output_winner(Board) :-
	check_victory('o', Board),
	write('Le joueur o a gagné !'), nl, !.

output_winner(_) :-
	write('Match nul !'), nl.

output_winner(Board, R) :-
    (   check_victory('x', Board) -> R = 'x'
    ;   check_victory('o', Board) -> R = 'o'
    ;   R = '_'
    ).
