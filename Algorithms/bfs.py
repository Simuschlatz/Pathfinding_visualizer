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
                    neighbor.make_closed(grid)
                    grid.visited.add(neighbor)

        # if node != start:
        #     node.make_closed(grid)
        draw()
    print(False)