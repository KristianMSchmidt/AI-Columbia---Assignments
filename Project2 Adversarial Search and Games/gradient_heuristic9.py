"""
Test 1: 1046. Men saa meget stabilt ud!
Test 2: 1046. Igen staert paa vej mod de 2046
Test 3:
"""

def heuristic(grid):
    if not grid.canMove():
        return - float('inf')

    ur = 0
    ul = 0
    lr = 0
    ll = 0

    for row in range(4):
        for col in range(4):

            cell = (row, col)
            val = grid.map[row][col]

            #ur += UR[cell]*val
            #ul += UL[cell]*val
            lr += LR[cell]*val
            #ll += LL[cell]*val

    monotonicity_score, max_c = max((ur, "UR"), (ul, "UL"), (lr, "LR"), (ll, "LL"))

    peer_neighbor_score = 0

    for row in range(4):
        for col in range(3):
            cell1 = (row, col)
            cell2 = (row, col+1)
            val1 = grid.map[row][col]
            val2 = grid.map[row][col+1]

            if val1 == val2:

                if max_c == "UR":
                    peer_neighbor_score +=  val1 * 0.5*max(UR[cell1], UR[cell2])

                elif max_c == "UL":
                    peer_neighbor_score +=  val1 * 0.5*max(UL[cell1], UL[cell2])

                elif max_c == "LR":
                    peer_neighbor_score +=  val1 * 0.5*max(LR[cell1], LR[cell2])

                elif max_c == "LL":
                    peer_neighbor_score +=  val1 * 0.5*max(LL[cell1], LL[cell2])


    for col in range(4):
        for row in range(3):
            val1 = grid.map[row][col]
            val2 = grid.map[row+1][col]

            if val1 == val2:

                if max_c == "UR":
                    peer_neighbor_score +=  val1 * 0.5*max(UR[cell1], UR[cell2])

                elif max_c == "UL":
                    peer_neighbor_score +=  val1 * 0.5*max(UL[cell1], UL[cell2])

                elif max_c == "LR":
                    peer_neighbor_score +=  val1 * 0.5*max(LR[cell1], LR[cell2])

                elif max_c == "LL":
                    peer_neighbor_score +=  val1 * 0.5*max(LL[cell1], LL[cell2])

    return monotonicity_score + peer_neighbor_score



# Weights
v1 = 64
v2 = 32
v3 = 16
v4 = 8
v5 = 4
v6 = 2
v7 = 1

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
