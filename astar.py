import math


class Tile:
    """A tile is a walkable space on a map."""

    distance = 0
    came_from = None

    def __init__(self, x, y, weight=1):
        self.x = x
        self.y = y
        self.weight = 1
        assert self.x is not None and self.y is not None

    def update_origin(self, came_from):
        """Update which tile this one came from."""
        self.came_from = came_from
        self.distance = came_from.distance + self.weight

    def __eq__(self, other):
        """A tile is the same if they have the same position"""
        return other and self.x == other.x and self.y == other.y

    def __lt__(self, other):
        """We want the shortest distance tile to find the happy path.
        This is used by min() so we can just compare them :)
        """
        return self.distance + self.weight <= other.distance

    def __hash__(self):
        """We need this so we can use a set()"""
        return hash(str(self))

    @property
    def pos(self):
        """a (x, y) tuple with position on the grid"""
        return (self.x, self.y)

    def __str__(self):
        return str(self.pos)

    def __repr__(self):
        return str(self)


class AStar:
    """The A Star (A*) path search algorithm"""

    def __init__(self, world, reward_world, target=(0, 0)):
        self.world = world
        self.reward_world = reward_world

        self.target_x, self.target_y = target

    def search(self, start_pos, target_pos):
        """A_Star (A*) path search algorithm"""

        self.target_x, self.target_y = target_pos

        start = Tile(*start_pos)
        self.open_tiles = set([start])
        self.closed_tiles = set()
        samples_count = 0

        # while we still have tiles to search
        while len(self.open_tiles) > 0:
            # get the tile with the shortest distance
            tile = min(self.open_tiles)
            tile_x = tile.x
            tile_y = tile.y
            # check if we're there. Happy path!
            if tile.pos == target_pos:
                return self.rebuild_path(tile), samples_count, self.reward_world
            # search new ways in the neighbor's tiles.
            self.search_for_tiles(tile)

            self.close_tile(tile)
            samples_count += 1
        # if we got here, path is blocked :(
        return None, samples_count, self.reward_world

    def search_for_tiles(self, current):
        """Search for new tiles in the maze"""
        for other in self.get_neighbors(current):
            if self.is_new_tile(other):
                other.update_origin(current)
                self.open_tiles.add(other)

            # if this other has gone a farthest distance before
            #   then we just found a new and shortest way to it.
            elif other > current:
                other.update_origin(current)
                if other in self.closed_tiles:
                    self.reopen_tile(other)

    def get_neighbors(self, tile):
        """Return a list of available tiles around a given tile"""
        min_x = max(0, tile.x - 1)
        max_x = min(len(self.world) - 1, tile.x + 1)
        min_y = max(0, tile.y - 1)
        max_y = min(len(self.world[tile.x]) - 1, tile.y + 1)
        reward_max = 0
        direction_max = []

        available_tiles = [
            (min_x, tile.y),
            (max_x, tile.y),
            (tile.x, min_y),
            (tile.x, max_y),
        ]
        neighbors = []
        for x, y in available_tiles:
            r1 = self.reward(x, y)
            if r1 > reward_max:
                reward_max = r1
                direction_max = x, y
            if (x, y) == tile.pos:
                continue

            if self.world[x][y] == 0:
                neighbors.append(Tile(x, y))

        self.reward_world[direction_max[0]][direction_max[1]] = reward_max

        return neighbors

    def reward(self, x1, y1):
        # print('reward', x1, y1)
        reward2 = -5
        try:
            reward2 = 1 / (math.sqrt((x1 - self.target_x) ** 2 + (y1 - self.target_y) ** 2))
        except:
            reward2 = 5
        # reward3 = 1 / reward2
        # print(reward2, reward3)
        return reward2 * 10

    def rebuild_path(self, current):
        """Rebuild the path from each tile"""
        self.last_tile = current
        path = []
        while current is not None:
            path.append(current)
            current = current.came_from
        path.reverse()
        # return a list with tuples
        return [tile.pos for tile in path]

    def is_new_tile(self, tile):
        """Check if this is a proviously unknown tile"""
        return tile not in self.open_tiles and tile not in self.closed_tiles

    def reopen_tile(self, tile):
        """Reinstate a tile in the open list"""
        self.closed_tiles.remove(tile)
        self.open_tiles.add(tile)

    def close_tile(self, tile):
        """Remove tile from open_tiles, as we are done testing it"""
        self.open_tiles.remove(tile)
        self.closed_tiles.add(tile)
