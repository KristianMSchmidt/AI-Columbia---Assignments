"""
Skal tjekkes igennem for bugs - f.eks. at den ikke returerneer None
"""
from random import randint
from BaseAI import BaseAI
import time

# Time Limit Before Losing
timeLimit = 0.2
allowance = 0.05


def heuristic(grid):
    """
    Returns heuristic value of node.
    """
    #num_zeros = len(getAvailableCells(self))
    return grid.getMaxTile()

class PlayerAI(BaseAI):

    def minimax(self, grid, depth, players_turn, first_move, start_time):
        """
        Returns (best_score, first_move)
        """
        #print "depth", depth

        #if time.clock() - start_time >  0.15 or depth == 4:
        #    return heuristic(grid), first_move

        if depth == 3:
            return heuristic(grid), first_move


        if players_turn: #This is the maximizing player
            moves = grid.getAvailableMoves()

            if moves == []:
                return heuristic(grid), first_move

            max_value = (- float('inf'), None)
            for move in moves:
                 child = grid.clone()
                 child.move(move)
                 if first_move == None:
                     value = self.minimax(child, depth + 1, False, move, start_time)
                 else:
                     value = self.minimax(child, depth + 1, False, first_move, start_time)
                 max_value = max(max_value, value)

            return max_value

        else: # Minimizing player
            cells = grid.getAvailableCells()
            min_value = (float('inf'), None)

            for cell in cells:
                child = grid.clone()
                child.setCellValue(cell, 2)
                value_2 = self.minimax(child, depth + 1, True, first_move, start_time)

                child = grid.clone()
                child.setCellValue(cell, 4)
                value_4 = self.minimax(child, depth + 1, True, first_move, start_time)

                min_value = min(min_value, value_2, value_4)

            return min_value


    def getMove(self, grid):
        """
        Should return 0,1,2 or 3, corresponding to "up", "down", "left" or "right".
        """
        start_time = time.clock()
        score, move = self.minimax(grid, 0, True, None, start_time)
        print "Time used:", time.clock() - start_time
        return move
        #moves = grid.getAvailableMoves()
        #return moves[randint(0, len(moves) - 1)] if moves else None


def test_minimax():
    from Grid import Grid
    grid = Grid()
    playerAI = PlayerAI()

    grid.map = [[4,12,14,16],
                [0,0,0,0],
                [0,0,0,0],
                [4,12,14,16]]
    for row in grid.map:
        print row

    print playerAI.getMove(grid)

if __name__ == "__main__":
    #test_getMinMove()
    test_minimax()
