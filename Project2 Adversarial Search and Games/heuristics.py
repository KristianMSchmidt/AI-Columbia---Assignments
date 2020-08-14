from math import log
from Grid import Grid


def heuristic(grid):
    """
    Returns heuristic value of node.
    """
    num_moves = grid.getAvailableMoves()
    #zeros = len(grid.getAvailableCells())
    mono = monotonicity(grid)
    smooth = smoothness_log_squared(grid)

    #THIS SEEMED PROMISSING. HOWEVER, I WOULD LIKE TO SEE MORE CLUSTERING  IN THE CORNERS.
    #GOT 1000P in first run
    return 0.7*mono + zeros + 0.004*smooth

def test_heuristics():
    from Grid import Grid
    grid = Grid()
    grid2 = Grid()
    grid3 = Grid()
    grid4 = Grid()

    grid.map = [[32,30,28,13],
                [30,25,20,12],
                [24,22,20,11],
                [22,21,20,10]]

    grid2.map = [[32,34,28,13],
                [30,25,20,12],
                [24,22,20,11],
                [22,21,20,10]]


    grid3.map = [[1000, 516,0,4],
                [0,2,8,0],
                [32,0,8,2],
                [8,4,8,0]]

    grid4.map = [[16, 13, 8, 0],
                [3,  13,  5, 0],
                [1 , 4,  1, 0],
                [1,  1, 0, 0]]


    print 0.5*monotonicity(grid2)
    print 0.004*smoothness_log_squared(grid2)

def monotonicity(grid):
    """
    This heuristic tries to ensure that the values of the tiles are all either increasing or decreasing along
    both the left/right and up/down directions. This heuristic alone captures the intuition that many others
    have mentioned, that higher valued tiles should be clustered in a corner. It will typically prevent smaller
    valued tiles from getting orphaned and will keep the board very organized, with smaller tiles cascading in
    and filling up into the larger tiles.
    """

    #for r in grid.map:
    #    print r

    def mono_count(row):
        count = 0
        for indx, val in enumerate(row[:-1]):
            for val2 in row[indx+1:]:
                if val < val2:
                    count += 1
                elif val > val2:
                    count -= 1
                #else:
                #    count += 1
        return count


    row_mono = sum(map(mono_count, grid.map))

    cols = [[grid.map[i][j] for i in range(4)] for j in range(4)]

    col_mono = sum(map(mono_count, cols))

    #print "row mono", row_mono
    #print "col mono", col_mono

    return abs(row_mono) + abs(col_mono)

def smoothness_log_squared(grid):
    """
    This heuristics calculated the sum of the absolute values of the LOG2 of the differences between adjacent tiles.
    """

    score = 0

    def scale(tile_val):
        if tile_val == 0:
            return 0
        else:
            return log(tile_val)/log(2)

    for row in range(4):
        row = grid.map[row]
        for col in range(3):
            val1 = scale(row[col])
            val2 = scale(row[col+1])
            score += (abs(val1*val2*(val1-val2)))**1.5

    for col in range(4):
        for row in range(3):
            val1 = scale(grid.map[row][col])
            val2 = scale(grid.map[row+1][col])
            score += (abs(val1*val2*(val1-val2)))**1.5

    return -score



if __name__ == "__main__":
    test_heuristics()
