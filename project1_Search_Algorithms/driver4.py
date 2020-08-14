"""
Your job in this assignment is to write driver.py, which solves any 8-puzzle board when given an
arbitrary starting configuration.

The program will be executed as follows:
$ python driver.py <method> <board>

The method argument will be one of the following. You need to implement all three of them:
bfs (Breadth-First Search)
dfs (Depth-First Search)
ast (A-Star Search)

The board argument will be a comma-separated list of integers containing no spaces. For example, to use the bread-first search strategy to solve the input board given by the starting configuration {0,8,7,6,5,4,3,2,1}, the program will be executed like so (with no spaces between commas):
$ python driver.py bfs 0,8,7,6,5,4,3,2,1

#This is an alternative version, where each node has a parent. In this version, the solutions_string are not stored
in each node (since they can become very long). Insted I only store the last move in each node - and only recover
the final solution string, when the goal state is achieved. This is more elegant - but not so much faster, it
seems.

Improvement ideas:
1) Would numpy board-representations with numpy arrays (or simply strings) make my program faster?
2) Just a detail: Turn "valid directions" into "get children"-method
3) Turn Puzzle-class in to game state class? Have a separate solver class?
"""

import sys

class Puzzle:
    """
    Class representation for the rectangular puzzle
    """
    def __init__(self, puzzle_height, puzzle_width, initial_grid=None, last_move = "", search_depth = 0,
                parent = None):
        """
        Initialize puzzle with default height and width
        Returns a Puzzle object
        """
        #self._solution_moves = solution_moves #this is the string of moves made so far in the attempt to solve puzzle
        self._last_move = last_move # Only counting moves that are part of solution proces
        self._search_depth = search_depth
        self._height = puzzle_height
        self._width = puzzle_width
        self._grid = [[col + puzzle_width * row
                       for col in range(self._width)]
                      for row in range(self._height)]
        self._parent = parent

        if initial_grid != None:
            for row in range(puzzle_height):
                for col in range(puzzle_width):
                    self._grid[row][col] = initial_grid[row][col]

    def __str__(self):
        """
        Generate string representaion for puzzle
        Returns a string
        """
        ans = ""
        for row in range(self._height):
            ans += str(self._grid[row])
            ans += "\n"
        return ans

    #####################################
    # GUI methods

    def get_height(self):
        """
        Getter for puzzle height
        Returns an integer
        """
        return self._height

    def get_width(self):
        """
        Getter for puzzle width
        Returns an integer
        """
        return self._width

    def get_number(self, row, col):
        """
        Getter for the number at tile position pos
        Returns an integer
        """
        return self._grid[row][col]

    def set_number(self, row, col, value):
        """
        Setter for the number at tile position pos
        """
        self._grid[row][col] = value

    def clone(self):
        """
        Make a copy of the puzzle to update during solving
        Returns a Puzzle object
        """
        new_puzzle = Puzzle(self._height, self._width, self._grid, self._last_move, self._search_depth, self._parent)
        return new_puzzle

    ########################################################
    # Core puzzle methods

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

    def update_puzzle(self, move_string):
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

    ##################################################################
    # Methods for puzzle solving

    def valid_directions(self):
        """
        Returns list of possible (and sound) directions to move in - in "UDLR" (up, down, left, right) order
        """
        zero_row, zero_col = self.current_position(0, 0)

        last_move = self._last_move

        valid_directions = []

        if zero_row > 0 and last_move != "d":
            valid_directions.append("u")
        if zero_row < self._height - 1 and last_move != "u":
            valid_directions.append("d")
        if zero_col > 0 and last_move != "r":
            valid_directions.append("l")
        if zero_col < self._width - 1 and last_move != "l":
            valid_directions.append("r")
        return valid_directions

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


    def solve_puzzle_dfs(self):
        """
        Solves the puzzle using dfs-search.
        """
        #initialize stack (I use lists)
        stack = [self]  #the stack is "the frontier"

        # Set of board positions that are already in frontier/queue or have been there earlier. I use
        # a set datastructure for fast lookups
        in_frontier_or_explored = set([str(self._grid)])

        num_expanded_nodes = 0  #Total number of nodes, that have been in focus of attention
                                #I.e. the nodes whose solution_state has been checked and whose
                                #children has been added to the stack
        max_search_depth = 0
        count = 0
        while stack:
            # Remove node from stack
            node = stack.pop()

            #Check if solution is found
            if node.is_solved():
                path_to_goal = node.recover_path()
                return path_to_goal, num_expanded_nodes, max_search_depth

            # Expand the node.
            #   To expand a given node, we generate successor nodes adjacent to the current node, and add them to the
            #   frontier set. Note that if these successor nodes are already in the frontier, or have already been
            #   visited, then they should not be added to the frontier again.
            num_expanded_nodes += 1

            valid_directions = node.valid_directions()

            #Push onto the stack in reverse-UDLR order; popping off results in UDLR order.
            for direction in reversed(valid_directions):
                child = node.clone()
                child.update_puzzle(direction)
                child._last_move = direction
                child._search_depth += 1
                child._parent = node
                c_grid_str = str(child._grid)
                if not c_grid_str in in_frontier_or_explored:
                     stack.append(child)
                     in_frontier_or_explored.add(c_grid_str)

            #Check current search depth:
            current_search_depth = node._search_depth + 1

            if current_search_depth > max_search_depth:
                max_search_depth = current_search_depth
                #print "current max search depth", max_search_depth

    def solve_puzzle_bfs(self):
        """
        Solves the puzzle using bfs-search.
        """
        import Queue as Q
        frontier = Q.Queue() #q.put(x), q.get_nowait()
        frontier.put(self)

        # Set of board positions that are already in frontier/queue or have been there earlier. I use
        # a set datastructure for fast lookups
        in_frontier_or_explored = set([str(self._grid)])

        num_expanded_nodes = 0  #Total number of nodes, that have been in focus of attention
                                #I.e. the nodes whose solution_state has been checked and whose
                                #children has been added to the queue
        max_search_depth = 0

        while not frontier.empty():
            #Remove node from queue
            node = frontier.get(False)

            #Check if solution is found
            if node.is_solved():
                path_to_goal = node.recover_path()
                return path_to_goal, num_expanded_nodes, max_search_depth

            # Expand the node.
            #   To expand a given node, we generate successor nodes adjacent to the current node, and add them to the
            #   frontier set. Note that if these successor nodes are already in the frontier, or have already been
            #   visited, then they should not be added to the frontier again.
            num_expanded_nodes += 1
            valid_directions = node.valid_directions()
            for direction in valid_directions:
                #Enqueue in UDLR order; dequeuing results in UDLR order
                child = node.clone()
                child.update_puzzle(direction)
                child._last_move = direction
                child._search_depth += 1
                child._parent = node
                c_grid_str = str(child._grid)
                if not c_grid_str in in_frontier_or_explored:
                     frontier.put(child)
                     in_frontier_or_explored.add(c_grid_str)


            #Check current search depth:
            current_search_depth = node._search_depth + 1

            if current_search_depth > max_search_depth:
                max_search_depth = current_search_depth

    def manhattan_dist(self):
        """
        Computes the total manhattan-distance between the given puzzle and the solved game state.
        """
        answer = 0

        #Calculate first row separately, as zero_tile should not be part of sum:
        for col in range(1, self._width):
            current_row, current_col = self.current_position(0,col)
            answer += abs(col-current_col) + current_row

        #Now the rest of the rows
        for row in range(1, self._height):
            for col in range(self._width):
                current_row, current_col = self.current_position(row,col)
                answer += abs(col-current_col) + abs(row - current_row)

        return answer

    def solve_puzzle_ast(self):
        """
        Solves the puzzle using A*search with Manhattan-distance heuristics.

        I add each child to the heap/frontier if it is not in the closed
        list (nb: this makes the heap longer)

        In case of ties, this version keeps the ULDR-order.
        """
        from heapq import heappush, heappop
        frontier = [] # The "frontier". I use heap to do fast extract minimums

        #to maintain UDLR-order in case of ties, we add 0 for u, 1 for d, 2 for l, 3 for r to tiil
        heappush(frontier, (self.manhattan_dist() + self._search_depth, 0, self))

        max_search_depth = 0

        closed = set()

        while frontier:
            #Remove node from heap
            _,_, node = heappop(frontier)

            #Check if solution is found
            if node.is_solved():
                path_to_goal = node.recover_path()
                num_expanded_nodes = len(closed) + 1
                return path_to_goal, num_expanded_nodes, max_search_depth

            #Add node to closed set and remove it from frontier_dict
            node_grid = str(node._grid)
            closed.add(node_grid)

            # Expand the node.
            #   To expand a given node, we generate successor nodes adjacent to the current node, and add them to the
            #   frontier set. Note that if these successor nodes are already in the frontier, or have already been
            #   visited, then they should not be added to the frontier again.
            valid_directions = node.valid_directions()
            for direction in valid_directions:
                child = node.clone()
                child.update_puzzle(direction)
                c_grid = str(child._grid)

                if c_grid in closed: #already evaluated
                    continue

                child._last_move = direction
                child._search_depth += 1
                child._parent = node

                if direction == "u": dir_priority = 0
                if direction == "d": dir_priority = 1
                if direction == "l": dir_priority = 2
                if direction == "r": dir_priority = 3

                #If the child is not in the closed list, if chosen to add it to the frontier
                #nomatter it is an interely new game state or if a similar game_state is already in the fronter
                # This is not wrong (I think), but it makes the heap a bit longer.
                # The suggested solution is to update the heap (using key_down-techniques),
                # if the game state is already there, but with a
                # higher search_depth
                heappush(frontier, (child.manhattan_dist() + child._search_depth, dir_priority, child))

            #Check current search depth:
            current_search_depth = node._search_depth + 1

            if current_search_depth > max_search_depth:
                max_search_depth = current_search_depth

    def solve_puzzle_gbfs(self):
        """
        Solves the puzzle using gready_best_first_search with Manhattan-heuristics.
        This is to find a short solution (not necessarily the shortest) in the minimal time.

        I add each child to the heap/frontier if it is not in the closed
        list (nb: this makes the heap longer)
        """

        from heapq import heappush, heappop
        frontier = [] # The "frontier". I use heap to do fast extract minimums

        heappush(frontier, (self.manhattan_dist(), self))

        max_search_depth = 0

        closed = set()

        while frontier:
            #Remove node from heap
            _, node = heappop(frontier)

            #Check if solution is found
            if node.is_solved():
                path_to_goal = node.recover_path()
                num_expanded_nodes = len(closed) + 1
                return path_to_goal, num_expanded_nodes, max_search_depth

            #Add node to closed set and remove it from frontier_dict
            node_grid = str(node._grid)
            closed.add(node_grid)

            # Expand the node.
            #   To expand a given node, we generate successor nodes adjacent to the current node, and add them to the
            #   frontier set. Note that if these successor nodes are already in the frontier, or have already been
            #   visited, then they should not be added to the frontier again.
            valid_directions = node.valid_directions()
            for direction in valid_directions:
                child = node.clone()
                child.update_puzzle(direction)
                c_grid = str(child._grid)

                if c_grid in closed: #already evaluated
                    continue

                child._last_move = direction
                child._search_depth += 1
                child._parent = node

                #If the child is not in the closed list, if chosen to add it to the frontier
                #nomatter it is an interely new game state or if a similar game_state is already in the fronter
                # This is not wrong (I think), but it makes the heap a bit longer.
                # The suggested solution is to update the heap (using key_down-techniques),
                # if the game state is already there, but with a
                # higher search_depth
                heappush(frontier, (child.manhattan_dist(), child))

            #Check current search depth:
            current_search_depth = node._search_depth + 1

            if current_search_depth > max_search_depth:
                max_search_depth = current_search_depth


    def solve_puzzle(self, method):
        """
        Takes a puzzle object and a method ("bfs", "dfs" or "ast" or "gbfs") and returns
        the triple: solution_string, num_expanded_nodes, max_search_depth
        """
        import time
        print "Solving below puzzle using {}-search:\n{}".format(method, self)

        start_time = time.time()
        if method == "bfs":
            solution_string, num_expanded_nodes, max_search_depth = self.solve_puzzle_bfs()
        elif method == "dfs":
            solution_string, num_expanded_nodes, max_search_depth = self.solve_puzzle_dfs()
        elif method == "ast":
            solution_string, num_expanded_nodes, max_search_depth = self.solve_puzzle_ast()
        elif method == "gbfs":
            solution_string, num_expanded_nodes, max_search_depth = self.solve_puzzle_gbfs()
        else:
            print "Unknown solution method"

        running_time = time.time() - start_time

        print "Search details:"
        print "Calculated solution string:'{}'".format(solution_string)
        print "Cost of path:", len(solution_string)
        print "Total number of expanded nodes", num_expanded_nodes
        print "Search depth:", len(solution_string)
        print "Max search depth:", max_search_depth
        print "Running time of search:", running_time, "seconds"
        import psutil
        print "Max_RAM_usage (in millions):",psutil.Process().memory_info().rss/float(1000000)
        #print "Expanded solution string:", convert_solution_string(solution_string)
        print ""
        print "Control of solution:"
        self.update_puzzle(solution_string)
        print "Puzzle after applying solution string:\n",self
        print "Puzzle is solved:", self.is_solved()

        return solution_string, num_expanded_nodes, max_search_depth


def convert_solution_string(sol_str):
    path_to_goal = []
    for index, letter in enumerate(sol_str):
        if letter == "u":
            path_to_goal.append("Up")
        elif letter == "d":
            path_to_goal.append("Down")
        elif letter == "l":
            path_to_goal.append("Left")
        else:
            path_to_goal.append("Right")
    return path_to_goal

def run_example(system_arguments):
    """
    Short cut to run 3x3-puzzles.
    """
    method = sys.argv[1]
    input_state = map(int, sys.argv[2].split(","))

    #initialize puzzle
    initial_state = [[0,0,0],[0,0,0],[0,0,0]]
    for row in range(3):
        for col in range(3):
            initial_state[row][col] = input_state[col + row*3]

    p = Puzzle(3, 3, initial_state)
    print p
    return p.solve_puzzle(method)

def run_random():
    p = Puzzle(3, 4)
    import random
    #random.seed(901)

    for i in range(1000):
        direction = random.choice(p.valid_directions())
        p.update_puzzle(direction)

    print p
    p_copy = p.clone()
    #p.solve_puzzle("ast")
    p_copy.solve_puzzle("gbfs")

def bfs_test_case1():
    sys.argv = ["path", "bfs", "6,1,8,4,0,2,7,3,5"]
    solution_string, num_expanded_nodes, max_search_depth =  run_example(sys.argv)

    solution_string = convert_solution_string(solution_string)
    print solution_string == ['Down', 'Right', 'Up', 'Up', 'Left', 'Down', 'Right', 'Down', 'Left', 'Up', 'Left', 'Up', 'Right', 'Right', 'Down', 'Down', 'Left', 'Left', 'Up', 'Up']
    print len(solution_string) == 20
    print num_expanded_nodes == 54094
    print max_search_depth == 21
def bfs_test_case2():
    sys.argv = ["path", "bfs", "8,6,4,2,1,3,5,7,0"]
    solution_string, num_expanded_nodes, max_search_depth =  run_example(sys.argv)

    solution_string = convert_solution_string(solution_string)
    print solution_string == ['Left', 'Up', 'Up', 'Left', 'Down', 'Right', 'Down', 'Left', 'Up', 'Right', 'Right', 'Up', 'Left', 'Left', 'Down', 'Right', 'Right', 'Up', 'Left', 'Down', 'Down', 'Right', 'Up', 'Left', 'Up', 'Left']
    print len(solution_string) == 26
    print num_expanded_nodes == 166786
    print max_search_depth == 27
def bfs_test_case3():
    sys.argv = ["path", "bfs", "1,2,5,3,4,0,6,7,8"]

    solution_string, num_expanded_nodes, max_search_depth =  run_example(sys.argv)

    solution_string = convert_solution_string(solution_string)
    print solution_string == ['Up', 'Left', 'Left']
    print len(solution_string) == 3
    print num_expanded_nodes == 10
    print max_search_depth == 4
def dfs_test_case0():
    sys.argv = ["path", "dfs", "3,1,2,6,4,5,0,7,8,9"]

    solution_string, num_expanded_nodes, max_search_depth =  run_example(sys.argv)

    solution_string = convert_solution_string(solution_string)
    print solution_string == ['Up', 'Up']
    print len(solution_string) == 2
    print num_expanded_nodes
    print max_search_depth
def dfs_test_case1():
    sys.argv = ["path", "dfs", "1,4,2,3,7,5,6,0,8"]

    solution_string, num_expanded_nodes, max_search_depth =  run_example(sys.argv)
    #print "Calculated solution string",solution_string
    solution_string = convert_solution_string(solution_string)
    print solution_string == ['Up', 'Up', "Left"]
    print len(solution_string) == 3
    print num_expanded_nodes
    print max_search_depth
def dfs_test_case2():
    sys.argv = ["path", "dfs", "3,2,5,6,1,8,7,4,0"]

    solution_string, num_expanded_nodes, max_search_depth =  run_example(sys.argv)

    solution_string = convert_solution_string(solution_string)
    print solution_string == ['Up', 'Up', "Left", "Down", "Down", "Left", "Up", "Up"]
    print len(solution_string) == 8
    print num_expanded_nodes
    print max_search_depth
def dfs_test_case3():
    sys.argv = ["path", "dfs", "8,6,4,2,1,3,5,7,0"]
    solution_string, num_expanded_nodes, max_search_depth =  run_example(sys.argv)

    print len(solution_string) == 9612
    print num_expanded_nodes == 9869
    print max_search_depth == 9612

    print "Correct path_to_goal is ['Up', 'Up', 'Left', ..., , 'Up', 'Up', 'Left']"
    #cost_of_path: 9612
    #nodes_expanded: 9869
    #search_depth: 9612
    #max_search_depth: 9612
def dfs_test_case4():
    sys.argv = ["path", "dfs", "6,1,8,4,0,2,7,3,5"]
    solution_string, num_expanded_nodes, max_search_depth =  run_example(sys.argv)

    print len(solution_string) == 46142
    print num_expanded_nodes == 51015
    print max_search_depth == 46142
    print "Correct path_to_goal: ['Up', 'Left', 'Down', ... , 'Up', 'Left', 'Up', 'Left']"
    # cost_of_path: 46142
    # nodes_expanded: 51015
    # search_depth: 46142
    # max_search_depth: 46142

def ast_test_case1():
    sys.argv = ["path", "ast", "6,1,8,4,0,2,7,3,5"]
    solution_string, num_expanded_nodes, max_search_depth =  run_example(sys.argv)

    print len(solution_string) == 20
    #num of xpanded notes shoudl be approx 696
    print max_search_depth == 20
    print ['Down', 'Right', 'Up', 'Up', 'Left', 'Down', 'Right', 'Down', 'Left', 'Up', 'Left', 'Up',
    'Right', 'Right', 'Down', 'Down', 'Left', 'Left', 'Up', 'Up'] == convert_solution_string(solution_string)

def ast_test_case2():
    sys.argv = ["path", "ast", "8,6,4,2,1,3,5,7,0"]
    solution_string, num_expanded_nodes, max_search_depth =  run_example(sys.argv)

    print len(solution_string) == 26
    #num_expanded_nodes should be approx: 1585
    print max_search_depth == 26
    print  ['Left', 'Up', 'Up', 'Left', 'Down', 'Right', 'Down', 'Left', 'Up', 'Right', 'Right', 'Up', 'Left', 'Left',
     'Down', 'Right', 'Right', 'Up', 'Left', 'Down', 'Down', 'Right',
      'Up', 'Left', 'Up', 'Left'] == convert_solution_string(solution_string)


def fifteen_puzzle():
    ##Joe Warrens challenge puzzle. My rigid tile by tile solution is about 236 moves. Optimal solution is about 80 moves.
    Joes_puzzle=Puzzle(4, 4, [[15, 11, 8, 12], [14, 10, 9, 13], [2, 6, 1, 4], [3, 7, 5, 0]])
    Joes_puzzle.solve_puzzle("gbfs")

    #p = Puzzle(4, 4)
    #import random
    #random.seed(901)
    #for i in range(40
    #):
    #    direction = random.choice(p.valid_directions())
    #    p.update_puzzle(direction)
    #p_copy = p.clone()
    #p.solve_puzzle("gbfs")




if __name__ == "__main__":
    #bfs_test_case1()
    #bfs_test_case2()
    #bfs_test_case3()
    #sys.argv = ["c:\krms...", "bfs", "0,8,7,6,5,4,3,2,1"]
    #run_example(sys.argv)
    #run_random()
    #dfs_test_case0()
    #dfs_test_case1()
    #dfs_test_case2()
    #dfs_test_case3()
    #dfs_test_case4()
    #fifteen_puzzle()
    #p = Puzzle(3,3)
    #p.update_puzzle("ddrru")
    #p.solve_puzzle("ast")

    #ast_test_case1()
    #ast_test_case2()

#Begin by writing a class to represent the state of the game at a given turn, including parent and child nodes. We suggest writing a separate solverclass to work with the state class. Feel free to experiment with your design, for example including a board class to represent the low-level physical configuration of the tiles, delegating the high-level functionality to the state class. When comparing your code with pseudocode, you might come up with another class for organising specific aspects of #your search algorithm elegantly.
