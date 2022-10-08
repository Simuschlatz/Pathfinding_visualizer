import pygame
import math
import random
from queue import PriorityQueue, Queue
from time import perf_counter
from Timer import timer
import collections

WIDTH, HEIGHT = 1000, 1000

WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pathfinding Algorithms visualization")

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 200, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
TURQUOISE = (64, 224, 208)
PURPLE = (128, 0, 128)
PINK = (250, 0, 250)
BLACK = (0, 0, 0)
DARK_BLUE = (0, 0, 70)
WHITE = (255, 255, 255)
GREY = (200, 200, 200)

 
ROWS = 50
NODE_WIDTH = HEIGHT // ROWS
COLS = WIDTH // NODE_WIDTH
print(ROWS, COLS)

Fps = 150
DELTA_SIZE_CHANGE = 15 / ROWS
class Node:
    def __init__(self, row, col, width, total_rows, total_cols):
        self.row = row
        self.col = col
        self.x = col * width
        self.y = row * width
        self.i_x = col * width
        self.i_y = row * width
        self.initial_width = width
        self.width = 0
        self.color = WHITE
        self.intended_color = None
        self.animate = False
        self.shrink = False
        self.increase = False
        self.neighbors = []
        self.update_adjacent_x2 = []
        self.total_rows = total_rows
        self.total_cols = total_cols
        self.visited = False

    def get_pos(self):
        return self.row, self.col

    def is_common(self):
        return self.color == WHITE

    def is_closed(self):
        return self.color == TURQUOISE

    def is_open(self):
        return self.color == BLUE

    def is_barrier(self):
        return self.color == DARK_BLUE

    def is_start(self):
        return self.color == ORANGE

    def is_end(self):
        return self.color == GREEN

    def update_pos(self):
        self.x = self.i_x + (self.initial_width - self.width) // 2
        self.y = self.i_y + (self.initial_width - self.width) // 2

    def prepare_animation_normal(func):
        def wrapper(self, grid, *args):
            grid.animating_hierarchy.remove(self)
            grid.animating_hierarchy.append(self)
            self.width = 4
            self.shrink = False
            self.animate = True
            self.increase = True
            return func(self, *args)
        return wrapper

    def increase_width(self):
        if self.width < self.initial_width + 8:
            self.width += DELTA_SIZE_CHANGE
            return
        self.increase = False

    def decrease_width(self):
        if self.width >= self.initial_width + DELTA_SIZE_CHANGE:
            self.width -= DELTA_SIZE_CHANGE
            return
        self.animate = False

    def play_animation_normal(self):
        if self.increase:
            self.increase_width()
        else:
            self.decrease_width()
        self.update_pos()

    def play_animation_shrink(self):
        if self.width >= DELTA_SIZE_CHANGE:
            self.width -= DELTA_SIZE_CHANGE
            self.update_pos()
            return
        self.animate = False
        self.shrink = False
        self.color = self.intended_color

    def change_color(self, color):
        self.animate = False
        self.width = self.initial_width
        self.update_pos()
        self.color = color

    def reset(self):
        self.animate = True
        self.shrink = True
        self.intended_color = WHITE

    def make_maze_generator(self):
        self.width = self.initial_width
        self.color = PINK

    @prepare_animation_normal
    def make_start(self):
        self.color = ORANGE

    @prepare_animation_normal
    def make_closed(self):
        self.color = TURQUOISE

    @prepare_animation_normal
    def make_open(self):
        self.color = BLUE

    @prepare_animation_normal
    def make_barrier(self):
        self.color = DARK_BLUE

    @prepare_animation_normal
    def make_end(self):
        self.color = GREEN

    @prepare_animation_normal
    def make_path(self):
        self.color = YELLOW

    def switch_barr_reset(self, grid):
        if self.is_barrier():
            self.color = WHITE
        else: self.make_barrier(grid)

    def draw(self, win):
        if self.animate:
            if self.shrink:
                self.play_animation_shrink()
            else: 
                self.play_animation_normal()
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))

    def shuffle_lst(lst):
        return random.shuffle(lst)
        
    def update_neighbors(self, grid):
        """""returns adjacent neighbors if neighbor is not a barrier
        """
        self.neighbors = []

        if self.row > 0 and not grid.nodes[self.row - 1][self.col].is_barrier():
            self.neighbors.append(grid.nodes[self.row - 1][self.col])
        if self.row < self.total_rows - 1 and not grid.nodes[self.row + 1][self.col].is_barrier():
            self.neighbors.append(grid.nodes[self.row + 1][self.col])
        if self.col > 0 and not grid.nodes[self.row][self.col - 1].is_barrier():
            self.neighbors.append(grid.nodes[self.row][self.col - 1])
        if self.col < self.total_cols - 1 and not grid.nodes[self.row][self.col + 1].is_barrier():
            self.neighbors.append(grid.nodes[self.row][self.col + 1])

    def __lt__():
        return False

class Grid:
    def __init__(self, rows, cols, NODE_WIDTH):
        """creates a two-dimensional grid of nodes
        """
        self.nodes = []
        self.animating_hierarchy = []
        self.prev_animating_hierarchy = []
        for i in range(rows):
            self.nodes.append([])
            for j in range(cols):
                node = Node(i, j, NODE_WIDTH, rows, cols)
                self.nodes[i].append(node)
                self.animating_hierarchy.append(node)
        self.start = None
        self.end = None
        self.visited = set()
        self.visited_cleared = True

    def draw(self, win, rows, cols, NODE_WIDTH):
        for node in self.animating_hierarchy:
            if node.is_common(): continue
            node.draw(win)
        # for i in range(cols):
        #     pygame.draw.line(win, GREY, (i * NODE_WIDTH, 0), (i * NODE_WIDTH, HEIGHT))
        # for i in range(rows):
        #     pygame.draw.line(win, GREY, (0, i * NODE_WIDTH), (WIDTH, i* NODE_WIDTH))

    def create_walls(self, grid, rows, cols):
        grid = self.nodes
        walls = set()
        for i in range(1, rows - 1):
            grid[i][0].switch_barr_reset(self)
            grid[i][cols - 1].switch_barr_reset(self)
        for i in range(cols):
            grid[0][i].switch_barr_reset(self)
            grid[rows - 1][i].switch_barr_reset(self)

    def clear(self, grid, rows, cols):
        grid = self.nodes
        for i in range(rows):
            for j in range(cols):
                node = grid[i][j]
                node.reset()

    def clear_visited(self):
        for node in self.visited:
            node.reset()
        self.start.make_start(self)
        self.end.make_end(self)
        self.visited = set()
        self.visited_cleared = True

    def fill_barr(self):
        for row in self.nodes:
            for node in row:
                if node == self.start or node == self.end:
                    continue
                node.change_color(DARK_BLUE)

def pre_algorithm_conditions(func):
    def wrapper(grid, *args):
        grid.clear_visited()
        grid.visited_cleared = False
        return func(grid, *args)
    return wrapper

def out_of_bnd(row, col):
    return row < 0 or row >= ROWS or col < 0 or col >= COLS

row, col = None, None
current_nodes = []  # List of cells [(x, y), ...]
last_nodes = []  # List of cells [(x, y), ...]
walking = True
range_ = list(range(4))
stack = []

def reset_dfs_vars():
    global current_nodes, last_nodes, walking
    current_nodes = []  # List of cells [(x, y), ...]
    last_nodes = []  # List of cells [(x, y), ...]
    walking = True

def walk_the_walk(grid, dir_one, dir_two, legal, rand):
    global row, col, walking, current_nodes
    if rand:
        random.shuffle(range_)
    for i in range_:
        temp_r, temp_c = dir_two[i](row, col)
        if not out_of_bnd(temp_r, temp_c):
            node = grid.nodes[temp_r][temp_c]
            if not node in grid.visited and legal(node) and node != grid.start:
                adj_row, adj_col = dir_one[i](row, col)
                adj_node = grid.nodes[adj_row][adj_col]
                current_nodes.append(adj_node)
                row, col = temp_r, temp_c
                grid.visited.add(node)
                walking = True
                return
    walking = False

def backtrack(grid, direction, legal):
    global row, col, walking, stack, finished
    row, col = stack.pop()
    if not stack:
        return
    for d in direction:
        temp_r, temp_c = d(row, col)
        if not out_of_bnd(temp_r, temp_c):
            node = grid.nodes[temp_r][temp_c]
            if not node in grid.visited and legal(node) and node != grid.start:
                walking = True
                return True

@pre_algorithm_conditions
def recursive_backtracking(grid, draw, maze, directions, legal_moves, rand):
    global row, col, stack, walking, current_nodes, last_nodes
    reset_dfs_vars()
    row, col = grid.start.get_pos()
    stack.append((row, col))
    if maze:
        grid.fill_barr()
    else:
        path = []
    """0 represents False, 1 represents True, so the condition
    statements will still work (if 1: ... and if True:... are the same)"""
    d_one, d_two = directions[maze]
    legal = legal_moves[maze]
    while stack:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        if walking:
            stack.append((row, col))
            walk_the_walk(grid, d_one, d_two, legal, rand)
        else:
            backtrack(grid, d_two, legal)
            if not maze:
                node.make_closed(grid)

        node = grid.nodes[row][col]

        if maze:
            current_nodes.append(node)
            for node in last_nodes:
                if node != grid.start and node != grid.end:
                    node.reset()
            for node in current_nodes:
                if node != grid.start and node != grid.end:
                    node.make_maze_generator()
            last_nodes, current_nodes = current_nodes, []
        else:
            node.make_open(grid)
            path.append(node)
            node.update_neighbors(grid)
            for neighbor in node.neighbors:
                if neighbor.is_end():
                    path = [grid.start] + list(filter(lambda node: node.is_open(), path))
                    path.append(grid.end)
                    animate_path(draw, grid, path, True)
                    stack = []
        draw()
    print(True)
    return True

def draw(win, grid, rows, cols, NODE_WIDTH):
	prev_t = perf_counter()
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			pygame.quit()
	win.fill(WHITE)

	grid.draw(win, rows, cols, NODE_WIDTH)
	pygame.display.update()

	current_t = perf_counter()
	Fps = 1 / (current_t - prev_t)
	print(Fps)
    
def mouse_pressed_idx(pos, NODE_WIDTH):
    mouseY, mouseX = pos
    row = mouseY // NODE_WIDTH
    col = mouseX // NODE_WIDTH
    return col, row

def swap(lst, pointer_1, pointer_2):
    store = lst[pointer_1]
    lst[pointer_1] = lst[pointer_2]
    lst[pointer_2] = store

def invert_lst(lst):
    pointer_1 = 0
    pointer_2 = len(lst) - 1

    while pointer_1 < pointer_2:
        swap(lst, pointer_1, pointer_2)
        pointer_1 += 1
        pointer_2 -= 1
    return lst

def animate_path(draw, grid, path, backwards):
    if not backwards:
        path = invert_lst(path) # Inverting the list this way is more time complex than looping backwards, but much fancier :)
    for i in range(len(path)):
        path[i].make_path(grid)
        draw()
    # Alternative (less fancy)
    """for i in range(len(path) - 1, -1, -1):       
        path[i].make_path(grid)
        draw()"""

@pre_algorithm_conditions
def breathfirst(grid, draw, start, backwards):
    queue = []
    visited = {start}
    queue.append((start, [start]))
    while queue:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        node, path = queue.pop()
        if node.is_end():
            backwards = not backwards
            animate_path(draw, grid, path, backwards)
            return True
            print(True)
        node.update_neighbors(grid)
        for neighbor in node.neighbors:
            if neighbor not in visited:
                new_path = path + [neighbor]
                queue = [(neighbor, new_path)] + queue
                visited.add(neighbor)

                if not neighbor.is_end():
                    neighbor.make_open(grid)
                    grid.visited.add(neighbor)

        if node != start:
            node.make_closed(grid)
        draw()
    print(False)

def heuristic(p1, p2):
    """returns 'Manhattan Distance' from p1 to p2
    """
    y1, x1 = p1
    y2, x2 = p2
    x_dist = x1 - x2
    y_dist = y1 - y2
    return abs(x_dist) + abs(y_dist)

def backtrack_path(draw, grid, origin, backwards):
    path = []
    node = grid.end
    path.append(node)
    while node in origin:
        node = origin[node]
        path.append(node)
    animate_path(draw, grid, path, backwards)

@pre_algorithm_conditions
def astar(grid, draw, start, end, backwards):
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
    
    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current_f, current_count, node = open_set.get()
        node.update_neighbors(grid)
        open_set_check.remove(node)

        if node == end:
            astar_path(draw, origin, start, end, backwards)
            return True

        for neighbor in node.neighbors:
            tem_g_score = g_score[node] + 1
            if tem_g_score < g_score[neighbor]:

                g_score[neighbor] = tem_g_score
                origin[neighbor] = node
                h_score = heuristic(neighbor.get_pos(), end.get_pos())
                f_score[neighbor] = tem_g_score + h_score

                if neighbor == end:
                    backtrack_path(draw, grid, origin, backwards)
                    return True

                if neighbor not in open_set_check:
                    count += 1
                    grid.visited.add(neighbor)
                    open_set.put((f_score[neighbor], count, neighbor))
                    open_set_check.add(neighbor)
                    neighbor.make_open(grid)
        if node != start:
            node.make_closed(grid)
        draw()

    return False
#---------------------------------------------------

def main(win):
    run = True
    search_started = False
    grid = Grid(ROWS, COLS, NODE_WIDTH)

    dir_one = [
        lambda row, col: (row, col + 1), # RIGHT
        lambda row, col: (row, col - 1), # LEFT
        lambda row, col: (row + 1, col), # DOWN
        lambda row, col: (row - 1, col), # UP
    ]

    dir_two = [
        lambda row, col: (row, col + 2), # RIGHT
        lambda row, col: (row, col - 2), # LEFT
        lambda row, col: (row + 2, col), # DOWN
        lambda row, col: (row - 2, col), # UP
    ]

    directions = [[dir_one, dir_one], [dir_one, dir_two]]

    legal_moves = [lambda node: not node.is_barrier(), lambda node: True]
    grid.clear(grid.nodes, ROWS, COLS)
    grid.start, grid.end = None, None
    while run:
        draw(win, grid, ROWS, COLS, NODE_WIDTH)
        # Manage events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if search_started:
                continue
            if pygame.mouse.get_pressed()[0]:
                mouse_pos = pygame.mouse.get_pos()
                row, col = mouse_pressed_idx(mouse_pos, NODE_WIDTH)
                node = grid.nodes[row][col]
                if not grid.start and node != grid.end:
                    grid.start = node
                    node.make_start(grid)

                elif not grid.end and node != grid.start:
                    grid.end = node
                    node.make_end(grid)
                elif node != grid.start and node != grid.end:
                    node.make_barrier(grid)
                    if node in grid.visited:
                        grid.visited.remove(node)
                    
            elif pygame.mouse.get_pressed()[2]:
                mouse_pos = pygame.mouse.get_pos()
                row, col = mouse_pressed_idx(mouse_pos, NODE_WIDTH)
                node = grid.nodes[row][col]

                if not grid.visited_cleared:
                    grid.clear_visited()
                    grid.visited_cleared = True
                if not node.is_common():
                    node.reset()
                    if node == grid.start:
                        grid.start = None
                    if node == grid.end:
                        grid.end = None
     
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    grid.clear(grid.nodes, ROWS, COLS)
                    grid.start, grid.end = None, None

                if event.key == pygame.K_w:
                    grid.create_walls(grid.nodes, ROWS, COLS)

                if event.key == pygame.K_SPACE and grid.start and grid.end:
                    recursive_backtracking(grid, lambda: draw(win, grid, ROWS, COLS, NODE_WIDTH), 1, directions, legal_moves, True)
                    recursive_backtracking(grid, lambda: draw(win, grid, ROWS, COLS, NODE_WIDTH), 0, directions, legal_moves, False)
                    breathfirst(grid, lambda: draw(win, grid, ROWS, COLS, NODE_WIDTH), grid.start, False)
                    astar(grid, lambda: draw(win, grid, ROWS, COLS, NODE_WIDTH), grid.start, grid.end, False)
    pygame.quit()

main(WIN)