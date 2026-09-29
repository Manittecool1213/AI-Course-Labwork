% Warehouse knowledge base and an independent verifier for proposed moves.

% Task 6: facts describing which locations are connected.
connected(a,b).
connected(b,a).
connected(b,c).
connected(c,b).

% The robot can move between connected locations.
% This is the Prolog form of the implication Connected(X, Y) -> CanMove(X, Y).
can_move(X,Y) :-
    connected(X,Y).

% Task 7: used to check a move proposed by the Python planner.
valid_move(X,Y) :-
    connected(X,Y).

% Extension: check a whole route, one move at a time.
valid_route([_]).
valid_route([X,Y|Rest]) :-
    valid_move(X,Y),
    valid_route([Y|Rest]).
