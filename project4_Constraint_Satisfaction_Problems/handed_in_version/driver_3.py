"""
Automatic soduko solver using two different algorithms:
 1) AC-3 algorithm (short for Arc Consistency Algorithm #3)
 2) Backtracking
I both cases, a soduko puzzle is viewed as a 'constraint satisfaction problem'.

This is the version I handed in.
"""

import sys
import copy

# List of sudoku tiles:
TILES = [row + col for row in "ABCDEFGHI" for col in "123456789"]

def BTS(sudoku):
    """
    Sudoku solver using Back Tracking Search (a sort of depth first search with "pruning").
    When a variable is assigned, I apply forward checking to reduce variables domains.
    """
    if is_solved(sudoku):
        return(sudoku)

    # Select unassigned variable (I use "Minimum remaining value" heuristic to do this)
    tile = min([(len(val), key) for key, val in sudoku.items() if len(val) > 1])[1]

    for value in sudoku[tile]: # I could speed up the solution by choosing order of values according to some heuristic.
        current_sudoku = copy.deepcopy(sudoku)  #This copy is not stricly necessary as long as AC3 does not modify it's argument. But I do it anyway, so that BTS does not alter arguments.
        current_sudoku[tile] = set([value])
        result_from_AC3 = AC_3(current_sudoku)
        if result_from_AC3 != False: # in this case AC_3 did not find any problems
            result = BTS(result_from_AC3)
            if result != False:
                return result
    return False

def AC_3(sudoku):
    """
    AC_3 solver for soduko.
    Returns false if an inconsistency is found. Returns simplified sudoku otherwise.
    NB: "Non-false" does not mean, that sudoku is solved or solvable.
    """
    constraints = CONSTRAINTS
    worklist = constraints.copy()
    current_sudoku = copy.deepcopy(sudoku)

    while worklist:
        X_i, X_j = worklist.pop()
        to_remove_from_X_i = revise(current_sudoku, constraints, X_i, X_j)
        if to_remove_from_X_i != None:
            current_sudoku[X_i].remove(to_remove_from_X_i)
            for constraint in constraints:
                if constraint[1] == X_i and constraint[0] != X_j:
                    worklist.add(constraint)

    for domain in current_sudoku.values():
        if not domain:
            return False

    return current_sudoku

def revise(sudoku, constraints, X_1, X_2):
    """
    Returns None if we are to do nothing to domain of X_1.
    Otherwise returns the value, that is to be removed from domain of X_1.
    """
    if (X_1, X_2) not in constraints or len(sudoku[X_2]) > 1:
        return None

    for val in sudoku[X_1]:
        if val in sudoku[X_2]:
            return val

    return None

def gen_constraints():
    # row and column constraints
    constraints = set([(tile_1, tile_2)
                  for tile_1 in TILES
                  for tile_2 in TILES
                  if (tile_1[0] == tile_2[0] or tile_1[1] == tile_2[1])
                  and tile_1 != tile_2])

    # square constraints:
    for ver in ["ABC", "DEF", "GHI"]:
        for hor in ["123","456", "789"]:
            square = [v+h for v in ver for h in hor]
            square_constraints = [(tile_1, tile_2)
                  for tile_1 in square
                  for tile_2 in square
                  if tile_1 != tile_2]
            constraints.update(square_constraints)

    return constraints

def gen_board(sudoku_string):
    sudoku = {}
    for indx, tile in enumerate(TILES):
        number = int(sudoku_string[indx])
        if number != 0:
            sudoku[tile] = set([number])
        else:
            sudoku[tile] = set(range(1,10))
    return sudoku

def is_solved(sudoku):
    """
    Checks if sudoku is solved (== only one option left at each position, and no inconsistencies)
    """
    for val in sudoku.values():
        if len(val) != 1:
            return False

    for tile_1, tile_2 in CONSTRAINTS:
        if sudoku[tile_1] == sudoku[tile_2]:
            return False

    return True

def gen_solve_string(sudoku):
    #assert(is_solved(sudoku))
    return "".join([str(next(iter(val))) for key, val in sorted(sudoku.items())])

def sudoku_solver(sudoku_string):

    sudoku = gen_board(sudoku_string)

    AC_3_attempt = AC_3(sudoku)

    if is_solved(AC_3_attempt):
        return gen_solve_string(AC_3_attempt) + " AC3"

    #BTS_solution = BTS(sudoku)
    BTS_solution = BTS(AC_3_attempt)

    return gen_solve_string(BTS_solution) + " BTS"

def test():

    with open("sudokus_finish.txt") as file:
        all_solutions = [line.strip() for line in file]

    with open("sudokus_start.txt") as all_sudokus:
        for i, sudoku in enumerate(all_sudokus):
            calculated = sudoku_solver(sudoku)
            expected = all_solutions[i]
            assert(calculated == expected)
            assert(is_solved(gen_board(calculated.split()[0])))
            print("#{}: {}".format(i+1, calculated))


# ======================= MISSION CONTROL  =====================================

CONSTRAINTS = gen_constraints()

#sys.argv = ["path", "000000000302540000050301070000000004409006005023054790000000050700810000080060009"]


if len(sys.argv) > 1:
   sudoku_string = sys.argv[1]
   solution = sudoku_solver(sudoku_string)
   fh = open('output.txt', 'w')
   fh.write(solution)
   fh.close()

else:
    test()
