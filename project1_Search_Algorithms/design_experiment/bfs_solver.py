import GameState

def solve_puzzle_dfs(puzzle):
    """
    Solves puzzle using dfs-search.
    """
    #initialize stack (I use lists)
    stack = [puzzle]  #the stack is "the frontier"

    # Set of board positions that are already in frontier/queue or have been there earlier. I use
    # a set datastructure for fast lookups
    in_frontier_or_explored = set([str(puzzle._grid)])

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

        #Push onto the stack in reverse-UDLR order; popping off results in UDLR order.
        node.set_children()
        for child in node._children:
            child_game_state = str(child._grid)
            child._search_depth += 1
            current_search_depth = child._search_depth

            if child_game_state in in_frontier_or_explored:
                continue

            stack.append(child)
            in_frontier_or_explored.add(child_game_state)


        if current_search_depth > max_search_depth:
            max_search_depth = current_search_depth

def main():
    p = GameState.GameState(3,3)
    p.update_puzzle("rrddlur")
    print p
    result = solve_puzzle_dfs(p)
    #print result
    path = result[0]
    p.update_puzzle(path)
    print p

if __name__ == "__main__":
    main()
