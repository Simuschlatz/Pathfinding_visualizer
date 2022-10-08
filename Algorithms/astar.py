from queue import PriorityQueue
import sys, os

def heuristic(p1, p2):
    """returns 'Manhattan Distance' from p1 to p2
    """
    y1, x1 = p1
    y2, x2 = p2
    x_dist = x1 - x2
    y_dist = y1 - y2
    return abs(x_dist) + abs(y_dist)

def backtrack_path(grid, origin, backwards):
    path = []
    node = grid.end
    path.append(node)
    while node in origin:
        node = origin[node]
        path.append(node)
    backwards = not backwards
    return path

def astar(grid, start, end, backwards):
    grid.clear_visited()
    grid.visited_cleared = False
    count = 0
    open_set = PriorityQueue()
    open_set_check = {start} # if node in open_set_check
    open_set.put((0, count, start))
    origin = {}
    # for row in grid: # No dict-comprehension beacuse two variables are declared
    #     for node in row:
    g_score = {node: float("inf") for row in grid.nodes for node in row}
    g_score[start] = 0
    f_score = {node: float("inf") for row in grid.nodes for node in row}
    f_score[start] = heuristic(start.get_pos(), end.get_pos())

    marked_nodes = []
    
    while not open_set.empty():

        current_f, current_count, node = open_set.get()
        node.update_neighbors(grid)
        open_set_check.remove(node)

        for neighbor in node.neighbors:
            tem_g_score = g_score[node] + 1
            if tem_g_score < g_score[neighbor]:

                g_score[neighbor] = tem_g_score
                origin[neighbor] = node
                h_score = heuristic(neighbor.get_pos(), end.get_pos())
                f_score[neighbor] = tem_g_score + h_score

                if neighbor == end:
                    path = backtrack_path(grid, origin, backwards)
                    return marked_nodes, path

                if neighbor not in open_set_check:
                    count += 1
                    grid.visited.add(neighbor)
                    open_set.put((f_score[neighbor], count, neighbor))
                    open_set_check.add(neighbor)
                    marked_nodes.append(neighbor)
    return False, False
