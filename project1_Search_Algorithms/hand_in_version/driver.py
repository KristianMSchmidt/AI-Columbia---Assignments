"""
Columbia AI. Project 1.
Kristian Moeller Schmdit, Copenhagen, Denmark
"""
import sys

class Puzzle:
    def __init__(self, puzzle_height, puzzle_width, initial_grid=None,
                last_move = "", search_depth = 0, parent = None):
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
        ans = ""
        for row in range(self._height):
            ans += str(self._grid[row])
            ans += "\n"
        return ans


    def clone(self):
        new_puzzle = Puzzle(self._height, self._width, self._grid, self._last_move, self._search_depth, self._parent)
        return new_puzzle


    def current_position(self, solved_row, solved_col):
        solved_value = (solved_col + self._width * solved_row)

        for row in range(self._height):
            for col in range(self._width):
                if self._grid[row][col] == solved_value:
                    return (row, col)
        assert False, "Value " + str(solved_value) + " not found"

    def update_puzzle(self, move_string):
        zero_row, zero_col = self.current_position(0, 0)
        for direction in move_string:
            if direction == "l":
                self._grid[zero_row][zero_col] = self._grid[zero_row][zero_col - 1]
                self._grid[zero_row][zero_col - 1] = 0
                zero_col -= 1
            elif direction == "r":
                self._grid[zero_row][zero_col] = self._grid[zero_row][zero_col + 1]
                self._grid[zero_row][zero_col + 1] = 0
                zero_col += 1
            elif direction == "u":
                self._grid[zero_row][zero_col] = self._grid[zero_row - 1][zero_col]
                self._grid[zero_row - 1][zero_col] = 0
                zero_row -= 1
            elif direction == "d":
                self._grid[zero_row][zero_col] = self._grid[zero_row + 1][zero_col]
                self._grid[zero_row + 1][zero_col] = 0
                zero_row += 1


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
                return path_to_goal, len(path_to_goal), num_expanded_nodes, max_search_depth

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
                child_game_state = str(child._grid)

                if child_game_state in in_frontier_or_explored:
                    continue

                child._last_move = direction
                child._search_depth += 1
                child._parent = node
                stack.append(child)
                in_frontier_or_explored.add(child_game_state)

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
                return path_to_goal, len(path_to_goal), num_expanded_nodes, max_search_depth

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
                child_game_state = str(child._grid)
                if child_game_state in in_frontier_or_explored:
                    continue
                child._last_move = direction
                child._search_depth += 1
                child._parent = node
                frontier.put(child)
                in_frontier_or_explored.add(child_game_state)


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
                return path_to_goal, len(path_to_goal), num_expanded_nodes, max_search_depth

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
                return path_to_goal, len(path_to_goal), num_expanded_nodes, max_search_depth

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


    def solve_puzzle(self, method, print_results = True):
        """
        Takes a puzzle object and a method ("bfs", "dfs" or "ast" or "gbfs") and returns
        the triple: solution_string, num_expanded_nodes, max_search_depth
        """
        import time
        if print_results:
            print "Solving below puzzle using {}-search:\n{}".format(method, self)

        start_time = time.time()
        if method == "bfs":
            solution_string, depth, num_expanded_nodes, max_search_depth = self.solve_puzzle_bfs()
        elif method == "dfs":
            solution_string, depth, num_expanded_nodes, max_search_depth = self.solve_puzzle_dfs()
        elif method == "ast":
            solution_string, depth, num_expanded_nodes, max_search_depth = self.solve_puzzle_ast()
        elif method == "gbfs":
            solution_string, depth, num_expanded_nodes, max_search_depth = self.solve_puzzle_gbfs()
        else:
            print "Unknown solution method"

        running_time = time.time() - start_time

        #import psutil
        #memory_usage =psutil.Process().memory_info().rss/float(1000000)

        if print_results:

            print "Search details:"
            #print "Calculated solution string:'{}'".format(solution_string)
            print "Cost of path:", len(solution_string)
            print "Total number of expanded nodes", num_expanded_nodes
            print "Search depth:", len(solution_string)
            print "Max search depth:", max_search_depth
            print "Running time of search:", running_time, "seconds"
            print "Max_RAM_usage (in millions):", memory_usage
            #print "Expanded solution string:", convert_solution_string(solution_string)
            print ""
            print "Control of solution:"
            self.update_puzzle(solution_string)
            print "Puzzle after applying solution string:\n",self
            print "Puzzle is solved:", self.is_solved()

        return solution_string, depth, num_expanded_nodes, max_search_depth, running_time, 0.00000000




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

def string_to_grid(grid_string):
    """
    Helper function to use when initial board is given as string.
    Assumes that puzzle has size 3x3.
    """
    input_state = map(int, grid_string.split(","))
    grid = [[0 for _ in range(3)] for _ in range(3)]

    for row in range(3):
        for col in range(3):
            grid[row][col] = input_state[col + row*3]
    return grid

def test_dfs():
    test_puzzle1 = Puzzle(3,3, string_to_grid("6,1,8,4,0,2,7,3,5"))
    test_puzzle2 = Puzzle(3,3, string_to_grid("8,6,4,2,1,3,5,7,0"))

    print ":::: Testing Depth First Search :::::"
    print "Test case 1"
    path, depth, num_expanded, max_depth = test_puzzle1.solve_puzzle_dfs()
    print "Calculated", depth, num_expanded, max_depth
    print "Expected", 46142, 51015, 46142
    test_puzzle1.update_puzzle(path)
    print "Puzzle solved?", test_puzzle1.is_solved()
    #print "Correct path_to_goal: ['Up', 'Left', 'Down', ... , 'Up', 'Left', 'Up', 'Left']"

    print "Test case 2"
    path, depth, num_expanded, max_depth = test_puzzle2.solve_puzzle_dfs()
    print "Calculated", depth, num_expanded, max_depth
    print "Expected", 9612, 9869, 9612
    test_puzzle2.update_puzzle(path)
    print "Puzzle solved?", test_puzzle2.is_solved()

    #print "Correct path_to_goal is ['Up', 'Up', 'Left', ..., , 'Up', 'Up', 'Left']"
def test_bfs():
    test_puzzle1 = Puzzle(3,3, string_to_grid("6,1,8,4,0,2,7,3,5"))
    test_puzzle2 = Puzzle(3,3, string_to_grid("8,6,4,2,1,3,5,7,0"))

    print ":::: Testing Bredth First Search :::::"
    print "Test case 1"
    path, depth, num_expanded, max_depth = test_puzzle1.solve_puzzle_bfs()
    print "Calculated", depth, num_expanded, max_depth
    print "Expected", 20, 54094, 21
    print "Correct path?", convert_solution_string(path) == ['Down', 'Right', 'Up', 'Up', 'Left', 'Down', 'Right', 'Down', 'Left', 'Up', 'Left', 'Up', 'Right', 'Right', 'Down', 'Down', 'Left', 'Left', 'Up', 'Up']

    print "Test case 2"
    path, depth, num_expanded, max_depth = test_puzzle2.solve_puzzle_bfs()
    print "Calculated", depth, num_expanded, max_depth
    print "Expected", 26, 166786, 27
    print "Correct path?", convert_solution_string(path) == ['Left', 'Up', 'Up', 'Left', 'Down', 'Right', 'Down', 'Left', 'Up', 'Right', 'Right', 'Up', 'Left', 'Left', 'Down', 'Right', 'Right', 'Up', 'Left', 'Down', 'Down', 'Right', 'Up', 'Left', 'Up', 'Left']
def test_ast():
    test_puzzle1 = Puzzle(3,3, string_to_grid("6,1,8,4,0,2,7,3,5"))
    test_puzzle2 = Puzzle(3,3, string_to_grid("8,6,4,2,1,3,5,7,0"))

    print ":::: Testing A* search :::::"
    print "Test case 1"
    path, depth, num_expanded, max_depth = test_puzzle1.solve_puzzle_ast()
    print "Calculated", depth, num_expanded, max_depth
    print "Expected", 20, "approx 696", 20
    print "Correct path?", convert_solution_string(path) == ['Down', 'Right', 'Up', 'Up', 'Left', 'Down', 'Right', 'Down', 'Left', 'Up', 'Left', 'Up',
    'Right', 'Right', 'Down', 'Down', 'Left', 'Left', 'Up', 'Up']
    print "Test case 2"
    path, depth, num_expanded, max_depth = test_puzzle2.solve_puzzle_ast()
    print "Calculated", depth, num_expanded, max_depth
    print "Expected", 26, "approx 1585", 26
    print "Correct path?", convert_solution_string(path) == ['Left', 'Up', 'Up', 'Left', 'Down', 'Right', 'Down', 'Left', 'Up', 'Right', 'Right', 'Up', 'Left', 'Left',
     'Down', 'Right', 'Right', 'Up', 'Left', 'Down', 'Down', 'Right',
      'Up', 'Left', 'Up', 'Left']

def run_random():
    import random
    p = Puzzle(3, 3)

    for i in range(1000):
        direction = random.choice(p.valid_directions())
        p.update_puzzle(direction)

    p_copy1 = p.clone()
    p_copy2 = p.clone()
    p_copy3 = p.clone()

    #p.solve_puzzle("dfs")
    #p_copy1.solve_puzzle("bfs")
    #p_copy2.solve_puzzle("ast")
    #p_copy3.solve_puzzle("gbfs")

def internal_run():

    #test_bfs()
    #test_dfs()
    test_ast()

    run_random()

    ##Joe Warrens challenge puzzle of size 4x4. Optimal solution is about 80 moves.
    #Joes_puzzle=Puzzle(4, 4, [[15, 11, 8, 12], [14, 10, 9, 13], [2, 6, 1, 4], [3, 7, 5, 0]])
    #Joes_puzzle.solve_puzzle("gbfs")

def solve_assignment():
    """
    This function is called, when script is run from command prompt with input of the kind:
    $ python driver.py <method> <board>
    Example given:
    $ python driver.py bfs 0,8,7,6,5,4,3,2,1
    It solves the puzzle with the specified search and writes data to the file output.txt
    """
    method = sys.argv[1]
    initial_grid = string_to_grid(sys.argv[2])
    P = Puzzle(3, 3, initial_grid)
    solution_string, cost, num_expanded_nodes, max_search_depth, running_time, max_ram_usage = P.solve_puzzle(method, False)
    running_time = "{0:.8f}".format(running_time)
    max_ram_usage = "{0:.8f}".format(max_ram_usage)

    fh = open('output.txt', 'w')
    fh.write("path_to_goal: {}\n".format(convert_solution_string(solution_string)))
    fh.write("cost_of_path: {}\n".format(cost))
    fh.write("nodes_expanded: {}\n".format(num_expanded_nodes))
    fh.write("search_depth: {}\n".format(cost))
    fh.write("max_search_depth: {}\n".format(max_search_depth))
    fh.write("running_time: {}\n".format(running_time))
    fh.write("max_ram_usage: {}\n".format(max_ram_usage))
    fh.close()

def run():
    """
    Determines if the program is being run to solve the assignment or as an internal test run.
    """
    if len(sys.argv) == 3:
        solve_assignment()

    else:
        internal_run()

################# Simulate run from command prompt
# Test case 1
#sys.argv = ["path", "bfs", "6,1,8,4,0,2,7,3,5"]
# Test case 2
#sys.argv = ["path", "ast", "8,6,4,2,1,3,5,7,0"]

run()
