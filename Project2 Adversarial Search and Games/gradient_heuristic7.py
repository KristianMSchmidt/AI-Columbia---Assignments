"""
DENNE ER AFLEVERET SOM FORSOEG 2!!!
Test1:2048
Test2: 1024
Test3: 1024
"""
def heuristic(grid):
    num_zeros = len(grid.getAvailableCells())

    ur = 0
    ul = 0
    lr = 0
    ll = 0

    for row in range(4):
        for col in range(4):

            cell = (row, col)
            val = grid.map[row][col]

            ur += UR[cell]*val
            ul += UL[cell]*val
            lr += LR[cell]*val
            ll += LL[cell]*val


    score = max(ur, ul, lr, ll)

    if num_zeros == 0:
        return score*0.4
    elif num_zeros == 1:
        return score*0.8
    elif num_zeros == 2:
        return score*0.9
    elif num_zeros == 3:
        return score*0.95
    elif num_zeros ==4:
        return score*0.98
    else:
        return score

# Weights
v1 = 14
v2 = 10
v3 = 6
v4 = 4
v5 = 2
v6 = 0.5
v7 = 0

# Gradient tables
UL = {(0,0):v1,    (0,1):v2,   (0,2):v3,    (0,3):v4,
      (1,0):v2,    (1,1):v3,   (1,2):v4,    (1,3):v5,
      (2,0):v3,    (2,1):v4,   (2,2):v5,    (2,3):v6,
      (3,0):v4,    (3,1):v5,   (3,2):v6,    (3,3):v7}


UR = {(0,0):v4,    (0,1):v3,   (0,2):v2,    (0,3):v1,
      (1,0):v5,    (1,1):v4,   (1,2):v3,    (1,3):v2,
      (2,0):v6,    (2,1):v5,   (2,2):v4,    (2,3):v3,
      (3,0):v7,    (3,1):v6,   (3,2):v5,    (3,3):v4}


LL = {(0,0):v4,    (0,1):v5,   (0,2):v6,    (0,3):v7,
      (1,0):v3,    (1,1):v4,   (1,2):v5,    (1,3):v6,
      (2,0):v2,    (2,1):v3,   (2,2):v4,    (2,3):v5,
      (3,0):v1,    (3,1):v2,   (3,2):v3,    (3,3):v4}


LR = {(0,0):v7,    (0,1):v6,   (0,2):v5,    (0,3):v4,
      (1,0):v6,    (1,1):v5,   (1,2):v4,    (1,3):v3,
      (2,0):v5,    (2,1):v4,   (2,2):v3,    (2,3):v2,
      (3,0):v4,    (3,1):v3,   (3,2):v2,    (3,3):v1}

if __name__ == "__main__":
    from Grid import Grid
    grid = Grid()
    grid.map[0][0] = 1
    grid.map[3][3] = 1
    print heuristic(grid)
