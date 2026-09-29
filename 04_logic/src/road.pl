% Task 8: chained inference from a fact through two rules.
wet_road.

slippery :-
    wet_road.

reduce_speed :-
    slippery.
