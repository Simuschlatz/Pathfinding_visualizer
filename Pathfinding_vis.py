import time
import pygame
import math
import random
from time import perf_counter
from Algorithms.astar import astar

WIDTH, HEIGHT = 1400, 1000

WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pathfinding Algorithms visualization")
pygame.font.init()

RED = [255, 0, 0]
RED_FINAL = [255, 100, 0]
GREEN = [0, 255, 0]
BLUE = [0, 200, 255]
YELLOW = [255, 255, 0]
ORANGE = (255, 165, 0)
TURQUOISE = (64, 224, 208)
PURPLE = (128, 0, 128)
PINK = (250, 0, 250)
BLACK = (0, 0, 0)
DARK_BLUE = (0, 0, 70)
WHITE = [255, 255, 255]
GREY = (200, 200, 200)
DARK_YELLOW = (235, 162, 52)
RED_YELLOW = (255, 108, 3)
 
ROWS = 50
NODE_WIDTH = HEIGHT // ROWS
COLS = WIDTH // NODE_WIDTH

UI_FONT = pygame.font.Font("freesansbold.ttf", 13)
# Initial size when animation is started
START_NODE_WIDTH = 4
POP_EFFECT = NODE_WIDTH / 4
ANIMATION_THRESHOLD = NODE_WIDTH + POP_EFFECT
TOTAL_ANIMATION_TIME = 0.5 # In seconds
# Difference in size during animation between each frame
DELTA_SIZE_CHANGE = 1
# TOTAL_ANIMATION_TIME * fps * DELTA_SIZE_CHANGE = ANIMATION_THRESHOLD
# <=> DELTA_SIZE_CHANGE = ANIMATION_THRESHOLD / TOTAL_ANIMATION_TIME / fps
fps = 4000

class Node:
    def __init__(self, row, col, width, total_rows, total_cols):
        self.row = row
        self.col = col
        self.modulate = False
        self.darken = False
        self.x = col * width
        self.y = row * width
        self.i_x = col * width
        self.i_y = row * width
        self.width = NODE_WIDTH
        self.color = WHITE
        self.intended_color = None
        self.animate = False
        self.shrink = False
        self.increase = False
        self.neighbors = []
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
        return self.color == DARK_YELLOW

    def is_end(self):
        return self.color == RED_YELLOW

    def update_pos(self):
        self.x = self.i_x + (NODE_WIDTH - self.width) // 2 + 1
        self.y = self.i_y + (NODE_WIDTH - self.width) // 2 + 1

    def update_delta_change(func):
        def wrapper(*args):
            global DELTA_SIZE_CHANGE
            # print(ANIMATION_THRESHOLD / TOTAL_ANIMATION_TIME / fps / DELTA_SIZE_CHANGE)
            result = func(*args)
            DELTA_SIZE_CHANGE = ANIMATION_THRESHOLD / TOTAL_ANIMATION_TIME / fps + 0.01
            print(DELTA_SIZE_CHANGE)
            # print(round(ANIMATION_THRESHOLD / DELTA_SIZE_CHANGE / TOTAL_ANIMATION_TIME)
            return result
        
        return wrapper

    def add_to_active_nodes(self, grid):
        if self in grid.active_nodes:
            grid.active_nodes.remove(self)
        grid.active_nodes.append(self)

    def prepare_animation_normal(func):
        def wrapper(self, grid, *args):
            self.add_to_active_nodes(grid)
            if self in grid.shrinking_nodes:
                grid.shrinking_nodes.remove(self)
            self.width = START_NODE_WIDTH
            self.shrink = False
            self.animate = True
            self.increase = True
            self.modulate = False
            result = func(self, *args)
            return result
        return wrapper

    def prepare_modulation(func):
        def wrapper(self, *args):
            self.modulate = True
            self.darken = True
            return func(self, *args)
        return wrapper

    def first_modulation(self):
        if self.color[1] < 240:
            self.color[1] += 1
            return
        self.darken = False

    def second_modulation(self):
        if self.color[1] > 100:
            self.color[1] -= 1
            return True
        self.modulate = False
        self.animate = False

    def increase_width(self):
        if self.width < ANIMATION_THRESHOLD:
            self.width += DELTA_SIZE_CHANGE
            return
        self.increase = False

    def decrease_width(self, grid, win):
        if self.width >= NODE_WIDTH:
            self.width -= DELTA_SIZE_CHANGE
            return
        if self in grid.active_nodes: grid.active_nodes.remove(self)
 

    def play_animation_normal(self, grid, win):
        if self.increase:
            self.increase_width()
        else:
            self.decrease_width(grid, win)
        self.update_pos() 

    def play_animation_shrink(self, win, grid):
        if self.width >= DELTA_SIZE_CHANGE:
            pygame.draw.rect(win, WHITE, (self.i_x, self.i_y, NODE_WIDTH, NODE_WIDTH))
            self.width -= DELTA_SIZE_CHANGE
            self.update_pos()
            return
        grid.shrinking_nodes.remove(self)
        self.animate = False
        self.shrink = False
        self.color = self.intended_color

    def change_color(self, color):
        self.animate = False
        self.shrink = False
        self.width = NODE_WIDTH
        self.update_pos()
        self.color = color

    def reset(self, grid):
        self.animate = True
        self.shrink = True
        self.intended_color = WHITE
        grid.shrinking_nodes.add(self)

    @update_delta_change
    def make_maze_generator(self):
        self.width = NODE_WIDTH
        self.color = PINK

    @prepare_animation_normal
    def make_start(self):
        self.color = DARK_YELLOW

    @update_delta_change
    @prepare_animation_normal
    @prepare_modulation
    def make_closed(self):
        self.color = list(RED)
        # self.modulate = True
        # self.darken = True

    # @update_delta_change
    # @prepare_animation_normal
    # def make_open(self):
    #     self.color = BLUE

    @update_delta_change
    @prepare_animation_normal
    def make_barrier(self):
        self.color = DARK_BLUE

    @prepare_animation_normal
    def make_end(self):
        self.color = RED_YELLOW

    @update_delta_change
    @prepare_animation_normal
    def make_path(self):
        self.color = YELLOW

    def switch_barr_reset(self, grid):
        if self.is_barrier():
            self.color = WHITE
        else: self.make_barrier(grid)

    # @update_delta_change
    def update(self, win, grid):
        global DELTA_SIZE_CHANGE
        if self.color == WHITE: return
        if self.animate:
            self.update_pos()
            if self.shrink:
                # pygame.draw.rect(win, RED, (self.i_x, self.i_y, NODE_WIDTH, NODE_WIDTH))
                self.play_animation_shrink(win, grid)
            else:
                self.play_animation_normal(grid, win)
        if self.modulate:
            if self.darken:
                self.first_modulation()
            else:
                self.second_modulation()
    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))

    def shuffle_lst(lst):
        return random.shuffle(lst)
        
    def update_neighbors(self, grid):
        """""returns adjacent neighbors if neighbor is not a barrier
        """
        self.neighbors = []

        if self.row > 0 and not grid.nodes[self.row - 1][self.col].is_barrier():
            self.neighbors.append(grid.nodes[self.row - 1][self.col])
        if self.row < ROWS - 1 and not grid.nodes[self.row + 1][self.col].is_barrier():
            self.neighbors.append(grid.nodes[self.row + 1][self.col])
        if self.col > 0 and not grid.nodes[self.row][self.col - 1].is_barrier():
            self.neighbors.append(grid.nodes[self.row][self.col - 1])
        if self.col < COLS - 1 and not grid.nodes[self.row][self.col + 1].is_barrier():
            self.neighbors.append(grid.nodes[self.row][self.col + 1])

    def get_surrounding_8(self, grid):
        surrounding_8 = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0: continue
                row, col = self.row + i, self.col + j
                if out_of_bnd(row, col): continue
                surrounding_8.append(grid.nodes[row][col])
        return surrounding_8

    def __lt__():
        return False
        
count = 0
class Grid:
    def __init__(self, rows, cols, NODE_WIDTH):
        """creates a two-dimensional grid of nodes
        """
        self.nodes = []
        self.surrounding_nodes = set()
        self.shrinking_nodes = set()
        self.active_nodes = []
        self.animating = []
        self.animating_hierarchy = []
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

    def draw(self, win, grid):
        global count
        # win.fill(WHITE)
        # print(len(self.active_nodes))
        if len(self.shrinking_nodes) > 0:
            print(count, len(self.shrinking_nodes))
        #     count += 1

        for i in range(5):
            self.animating_hierarchy[i].draw(win)

        self.surrounding_nodes.clear()
        for node in self.active_nodes:
            if not node.increase:
                for adj in node.get_surrounding_8(self):
                    self.surrounding_nodes.add(adj)
        self.surrounding_nodes -= set(self.active_nodes)
        for node in list(self.shrinking_nodes):
            node.update(win, grid)
            node.draw(win)
        for node in self.surrounding_nodes:
            node.draw(win) 

        # print(self.shrinking_nodes)
        for node in self.active_nodes:
            node.update(win, grid)
            node.draw(win)

        # for i in range(5, len(self.animating_hierarchy)):
        #     if self.animating_hierarchy[i].is_common(): continue
        #     self.animating_hierarchy[i].update(self, win)
        #     # node.draw(win)

        # for i in range(COLS):
        #     pygame.draw.line(win, GREY, (i * NODE_WIDTH, 0), (i * NODE_WIDTH, HEIGHT))
        # for i in range(ROWS):
        #     pygame.draw.line(win, GREY, (0, i * NODE_WIDTH), (WIDTH, i* NODE_WIDTH))

    def create_walls(self, grid, rows, cols):
        grid = self.nodes
        for i in range(1, rows - 1):
            grid[i][0].switch_barr_reset(self)
            grid[i][cols - 1].switch_barr_reset(self)
        for i in range(cols):
            grid[0][i].switch_barr_reset(self)
            grid[rows - 1][i].switch_barr_reset(self)

    def clear(self, grid, rows, cols):
        for node in self.animating_hierarchy:
            if node.is_common: continue
            node.reset(self)

    def clear_visited(self):
        for node in self.visited:
            node.reset(self)
            self.shrinking_nodes.add(node)
        self.start.make_start(self)
        self.end.make_end(self)
        self.visited = set()
        self.visited_cleared = True

    def fill_barr(self):
        for node in self.animating_hierarchy:
            if node == self.start or node == self.end: continue
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
def recursive_backtracking(grid, draw, win, maze, directions, legal_moves, rand, backwards = False):
    global row, col, stack, walking, current_nodes, last_nodes
    reset_dfs_vars()
    row, col = grid.start.get_pos()
    stack.append((row, col))
    if maze:
        win.fill(DARK_BLUE)
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
                    node.reset(grid)
            for node in current_nodes:
                if node != grid.start and node != grid.end:
                    node.make_maze_generator()
            last_nodes, current_nodes = current_nodes, []
        else:
            node.make_closed(grid)
            path.append(node)
            node.update_neighbors(grid)
            # for neighbor in node.neighbors:
            #     if neighbor.is_end():
            if node.is_end():
                path = [grid.start] + list(filter(lambda node: node.is_open(), path))
                path.append(grid.end)
                animate_path(draw, grid, path, backwards)
                stack = []
        draw()
    print(True)
    return True

def draw(win, grid):
    global fps
    prev_t = perf_counter()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
    grid.draw(win, grid)
    current_t = perf_counter()
    dt = current_t - prev_t
    fps = 1 / dt
    text_surface = UI_FONT.render(f"FPS: {int(fps)}", True, (0, 255, 0))
    # print(fps)
    win.blit(text_surface, (10, 6))

    pygame.display.update()
    
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
    if backwards:
        path = invert_lst(path) # Inverting the list this way is more time complex than looping backwards, but much fancier :)
    for i in range(len(path)):
        path[i].make_path(grid)
        if draw:
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

        node.update_neighbors(grid)
        for neighbor in node.neighbors:
            if neighbor not in visited:
                new_path = path + [neighbor]
                queue += [(neighbor, new_path)]
                visited.add(neighbor)
                if neighbor.is_end():
                    backwards = backwards
                    animate_path(draw, grid, path, backwards)
                    return True
                else:
                    neighbor.make_closed(grid)
                    grid.visited.add(neighbor)

        # if node != start:
        #     node.make_closed(grid)
        draw()
    print(False)

def visualize_search(draw, grid, marked_nodes, path):
    for node in marked_nodes:
        node.make_closed(grid)
        draw()
        # time.sleep(0.01)
    for node in path:
        node.make_path(grid)
        draw()

def instant_search(grid, marked_nodes, path):
    for node in marked_nodes:
        node.change_color(RED_FINAL)
    for node in path:
        node.change_color(YELLOW)

def main(win):
    global Fps
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

    dragging_start, dragging_end = False, False
    prev_start = None
    prev_end = None
    grid.start, grid.end = grid.nodes[ROWS // 2][COLS // 4], grid.nodes[ROWS // 2][COLS // 4 * 3]
    grid.start.make_start(grid)
    grid.end.make_end(grid)
    
    win.fill(WHITE)
    while run:
        draw(win, grid)
        # Manage events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.MOUSEBUTTONUP:
                dragging_start, dragging_end = False, False
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

                # elif node.is_barrier():
                #     node.reset(grid)
                # elif node == grid.start and dragging:
                #     prev_start.reset()
                #     node.make_start(grid)
                #     prev_start = node
                elif node != grid.start and node != grid.end and not node.is_barrier():
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

                if dragging_start and node != grid.end:
                    prev_start.reset(grid)
                    node.make_start(grid)
                    prev_start, grid.start = node, node
                    marked_nodes, path = astar(grid, grid.start, grid.end, False)
                    instant_search(grid, marked_nodes, path)
                elif node == grid.start:
                    dragging_start = True
                    dragging_end = False
                    prev_start = node
                elif dragging_end and node != grid.start:
                    prev_end.reset(grid)
                    node.make_end(grid)
                    prev_end, grid.end = node, node
                    marked_nodes, path = astar(grid, grid.start, grid.end, False)
                    instant_search(grid, marked_nodes, path)
                elif node == grid.end:
                    dragging_end = True
                    dragging_start = False
                    prev_end = node
                elif node.is_barrier():
                    node.reset(grid)
                    # if node == grid.start:
                    #     grid.start = None
                    # if node == grid.end:
                    #     grid.end = None
     
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    grid.clear(grid, ROWS, COLS)
                    grid.start, grid.end = None, None

                if event.key == pygame.K_w:
                    grid.create_walls(grid.nodes, ROWS, COLS)

                if event.key == pygame.K_SPACE and grid.start and grid.end:
                    #grid.fill_barr()
                    #recursive_backtracking(grid, lambda: draw(win, grid), win, 1, directions, legal_moves, True)
                    #recursive_backtracking(grid, lambda: draw(win, grid), win, 1, directions, legal_moves, False)
                    recursive_backtracking(grid, lambda: draw(win, grid), win, 0, directions, legal_moves, True)
                    #breathfirst(grid, lambda: draw(win, grid), grid.start, False)
                    marked_nodes, path = astar(grid, grid.start, grid.end, False)
                    # print(len(marked_nodes))
                    visualize_search(lambda: draw(win, grid), grid, marked_nodes, path)

    pygame.quit()

main(WIN)