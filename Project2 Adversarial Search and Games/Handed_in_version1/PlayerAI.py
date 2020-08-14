"""
PlayerAI for the 2048 game.
"""
from BaseAI import BaseAI
from heuristics import heuristic
import time

# Time Limit Before Losing
timeLimit = 0.2
allowance = 0.05


class PlayerAI(BaseAI):

    def minimax_alpha_beta_IDDFS(self, grid):
        """
        Minimax with alpha-beta-pruning. Iterative deepening depth-first-search.
        """
        start_time = time.clock()
        best_move = None

        for depth in xrange(1000):
            try:
                alpha = -float('inf')   #Best choice for max so far in search
                beta = float('inf')     #Best choice for min so far in search
                best_score, best_move = self.minimax_alpha_beta_DLS(grid, depth, True, None, start_time, alpha, beta)
                obtained_depth = depth

            except:
                print "Obtained search depth:", obtained_depth
                print "Time used:", time.clock() - start_time
                return best_move


    def minimax_alpha_beta_DLS(self, grid, depth, players_turn, first_move, start_time, alpha, beta):
        """
        Recursive Depth-Limited minimax search with alpha-beta-pruning.
        Returns (best_score, best_move) found, given the specified depth.
        """
        if time.clock() - start_time >  0.2:
            print "Time's up!"
            raise Exception("Time exeption")

        if depth == 0:
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
                    value = self.minimax_alpha_beta_DLS(child, depth - 1, False, move, start_time, alpha, beta)
                else:
                    value = self.minimax_alpha_beta_DLS(child, depth - 1, False, first_move, start_time, alpha, beta)
                max_value = max(max_value, value)
                alpha = max(alpha, max_value[0])
                if beta <= alpha:
                    #print "Beta cuf off!"
                    break #cut of branch

            return max_value

        else: # Minimizing player
            cells = grid.getAvailableCells()
            min_value = (float('inf'), None)

            for cell in cells:
                child = grid.clone()
                child.setCellValue(cell, 2)
                value_2 = self.minimax_alpha_beta_DLS(child, depth - 1, True, first_move, start_time, alpha,beta)

                child = grid.clone()
                child.setCellValue(cell, 4)
                value_4 = self.minimax_alpha_beta_DLS(child, depth - 1, True, first_move, start_time, alpha,beta)

                min_value = min(min_value, value_2, value_4)

                beta = min(beta, min_value[0])
                if beta <= alpha:
                    #print "Alpha cuf off!"
                    break #cut of branch

            return min_value

    def getMove(self, grid):
        """
        Should return 0,1,2 or 3, corresponding to "up", "down", "left" or "right".
        """
        move = self.minimax_alpha_beta_IDDFS(grid)
        return move

if __name__ == "__main__":
    pass
