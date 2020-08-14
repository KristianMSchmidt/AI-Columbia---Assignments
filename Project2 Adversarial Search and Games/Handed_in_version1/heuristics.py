
from math import log

def heuristic(grid):
    """
    Returns heuristic value of node.
    """
    zeros = len(grid.getAvailableCells())
    moves = num_moves(grid)
    mono = monotonicity(grid)
    smooth = smoothness(grid)

    return 0.6*mono + 2*moves + smooth + zeros


def num_moves(grid):

    num_moves = len(grid.getAvailableMoves())
    if num_moves:
        return num_moves
    else:
        return -float('inf')


def monotonicity(grid):

    def mono_count(row):
        count = 0
        for indx, val in enumerate(row[:-1]):
            for val2 in row[indx+1:]:
                if val < val2:
                    count += 1
                elif val > val2:
                    count -= 1
        return count


    row_mono = sum(map(mono_count, grid.map))
    cols = [[grid.map[i][j] for i in range(4)] for j in range(4)]
    col_mono = sum(map(mono_count, cols))

    return abs(row_mono) + abs(col_mono)

def smoothness(grid):

    score = 0

    for row in range(4):
        row = grid.map[row]
        for col in range(3):
            val1 = row[col]
            val2 = row[col+1]
            if 0 != val1 == val2:
                score += log(val1)/log(2)

    for col in range(4):
        for row in range(3):
            val1 = grid.map[row][col]
            val2 = grid.map[row+1][col]
            if 0 != val1 == val2:
                score += log(val1)/log(2)
    return score
