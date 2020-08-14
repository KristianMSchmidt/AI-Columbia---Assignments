"""
Design experiment. I actually like this design.
"""

class GameState:
    """
    Initialize puzzle with default height and width.
    Puzzle can be initialized with either a string or grid representation
    Returns a Puzzle object
    """
    def __init__(self, puzzle_height, puzzle_width, initial_grid = None,
                last_move = "", search_depth = 0, parent = None):
        self._last_move = last_move # Only counting moves that are part of solution proces
        self._search_depth = search_depth
        self._height = puzzle_height
        self._width = puzzle_width
        self._parent = parent
        self._grid = [[col + puzzle_width * row
                       for col in range(self._width)]
                      for row in range(self._height)]
        if initial_grid != None:
           for row in range(puzzle_height):
               for col in range(puzzle_width):
                   self._grid[row][col] = initial_grid[row][col]

    def set_children(self):
        self._children = self.generate_children()

    def __str__(self):
        """
        String representaion for puzzle
        """
        ans = ""
        for row in range(self._height):
            ans += str(self._grid[row])
            ans += "\n"
        return ans

    def clone(self):
        """
        Make a copy of the puzzle to update during solving
        Returns a Puzzle object
        """
        new_puzzle = GameState(self._height, self._width, self._grid, self._last_move, self._search_depth, self._parent)
        return new_puzzle


    def current_position(self, solved_row, solved_col):
        """
        Locate the current position of the tile that will be at
        position (solved_row, solved_col) when the puzzle is solved
        Returns a tuple of two integers
        """
        solved_value = (solved_col + self._width * solved_row)

        for row in range(self._height):
            for col in range(self._width):
                if self._grid[row][col] == solved_value:
                    return (row, col)
        assert False, "Value " + str(solved_value) + " not found"

    def update_puzzle(self, move_string, register_last_move = False):
        """
        Updates the puzzle state based on the provided move string
        """
        zero_row, zero_col = self.current_position(0, 0)
        for direction in move_string:
            if direction == "l":
                assert zero_col > 0, "move off grid: " + direction
                self._grid[zero_row][zero_col] = self._grid[zero_row][zero_col - 1]
                self._grid[zero_row][zero_col - 1] = 0
                zero_col -= 1
            elif direction == "r":
                assert zero_col < self._width - 1, "move off grid: " + direction
                self._grid[zero_row][zero_col] = self._grid[zero_row][zero_col + 1]
                self._grid[zero_row][zero_col + 1] = 0
                zero_col += 1
            elif direction == "u":
                assert zero_row > 0, "move off grid: " + direction
                self._grid[zero_row][zero_col] = self._grid[zero_row - 1][zero_col]
                self._grid[zero_row - 1][zero_col] = 0
                zero_row -= 1
            elif direction == "d":
                assert zero_row < self._height - 1, "move off grid: " + direction
                self._grid[zero_row][zero_col] = self._grid[zero_row + 1][zero_col]
                self._grid[zero_row + 1][zero_col] = 0
                zero_row += 1
            else:
                assert False, "invalid direction: " + direction

            if register_last_move:
                self._last_move = direction

    def generate_children(self):
        """
        Returns list of possible (and sound) directions to move in - in "UDLR" (up, down, left, right) order
        """
        zero_row, zero_col = self.current_position(0, 0)

        last_move = self._last_move
        children = []
        if zero_row > 0 and last_move != "d":
            u_child = self.clone()
            u_child.update_puzzle("u", True)
            u_child._parent = self
            children.append(u_child)
        if zero_row < self._height - 1 and last_move != "u":
            d_child = self.clone()
            d_child.update_puzzle("d", True)
            d_child._parent = self
            children.append(d_child)
        if zero_col > 0 and last_move != "r":
            l_child = self.clone()
            l_child.update_puzzle("l", True)
            l_child._parent = self
            children.append(l_child)
        if zero_col < self._width - 1 and last_move != "l":
            r_child = self.clone()
            r_child.update_puzzle("r", True)
            r_child._parent = self
            children.append(r_child)
        return children

    def recover_path(self):
        """
        Recovers path takes from startnode to the node in question.
        """
        reverse_path_to_goal = ""
        node = self
        while node._last_move != "":
            reverse_path_to_goal += node._last_move
            node = node._parent
        path_to_goal = reverse_path_to_goal[::-1] #reverse order
        return path_to_goal

    def is_solved(self):
        """
        Cheks is self is the desired goal_state
        """
        #This is the general code that works for all grid sizes:
        for row in range(self._height):
            for col in range(self._width):
               if self._grid[row][col] != col + self._width * row:
                   return False
        return True
