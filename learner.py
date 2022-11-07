import math
import random
import numpy as np
from colorama import Fore, Back, Style
import time
import pyastar2d


# https://www.geeksforgeeks.org/ml-monte-carlo-tree-search-mcts/


class World:
    def __init__(self):
        self.epsilon = 0.2
        self.dim_x = 12
        self.dim_y = 12

        self.arr1 = np.zeros([self.dim_x, self.dim_y], dtype=np.float32)
        self.arr2 = np.zeros([self.dim_x, self.dim_y])

        self.pos_x = 0
        self.pos_y = 0

        self.target_x = 11
        self.target_y = 1

        self.arr1[self.pos_x][self.pos_y] = 8
        self.arr1[self.target_x][self.target_y] = 3

        self.directions = [[0, 0], [0, 1], [0, -1], [1, 0], [1, 1], [1, -1], [-1, 0], [-1, 1], [-1, -1]]

        self.arr1[0][5] = 1
        self.arr1[1][5] = 1
        self.arr1[2][5] = 1
        self.arr1[3][5] = 1
        self.arr1[7][0] = 1
        self.arr1[7][1] = 1
        self.arr1[7][2] = 1
        self.arr1[7][3] = 1
        self.arr1[7][4] = 1
        self.arr1[7][5] = 1
        self.arr1[7][6] = 1
        self.arr1[7][7] = 1
        self.arr1[7][8] = 1
        self.arr1[6][8] = 1
        self.arr1[5][8] = 1
        self.arr1[4][8] = 1

        self.leaves = []

        # arr1[target_x][target_y] = 5

    def a_star_cost_array(self):
        temp_arr = np.copy(self.arr1)
        self.arr1[self.arr1 == 1] = 100
        self.arr1[self.arr1 == 0] = 1
        cost_arr = self.arr1
        self.arr1 = np.copy(temp_arr)
        return cost_arr

    def a_star(self):
        cost_arr = self.a_star_cost_array()
        start = (self.pos_x, self.pos_y)
        goal = (self.target_x, self.target_y)
        path = pyastar2d.astar_path(cost_arr, start, goal, allow_diagonal=True)

        # The path is returned as a numpy array of (i, j) coordinates.
        # print(f"Shortest path from {start} to {goal} found:")

        for item in path:
            self.arr1[item[0]][item[1]] = 8

        self.arr1[start[0]][start[1]] = 3
        self.arr1[goal[0]][goal[1]] = 3

    def position_move(self):
        pos_directions = []
        for direction in self.directions:
            ftr_x = self.pos_x + direction[0]
            ftr_y = self.pos_y + direction[1]

            if ftr_x >= 0 and ftr_y >= 0 and ftr_x < self.dim_x and ftr_y < self.dim_y:
                if self.arr1[ftr_x][ftr_y] != 1:
                    pos_directions.append((ftr_x, ftr_y))
        return pos_directions

    def move_predict(self):

        reward_max = -1
        best_next_position = []
        print("predicting, current pos", self.pos_x, self.pos_y)

        next_positions = self.position_move()
        # print("next position", next_positions)

        for position in next_positions:
            x1 = position[0]
            y1 = position[1]

            if self.arr1[x1][y1] == 0:
                r1 = self.reward(x1, y1)
                print("reward", r1)
                if r1 > reward_max:
                    reward_max = r1
                    best_next_position = position

        print("position chosen", best_next_position)
        print("")
        self.pos_x = best_next_position[0]
        self.pos_y = best_next_position[1]
        self.arr1[self.pos_x][self.pos_y] = 8
        self.arr2[x1][y1] = reward_max

    def move_explore(self):
        # pass
        reward_max = 0
        position_max = []
        rnd_positions = []
        print("exploring, current pos", self.pos_x, self.pos_y)
        rnd_positions = self.position_move()
        print(rnd_positions)

        rnd2 = random.randint(0, len(rnd_positions))
        position_max = rnd_positions[rnd2]
        print("position chosen", position_max)
        print("")
        self.pos_x = position_max[0]
        self.pos_y = position_max[1]
        print("########################")
        print(self.pos_x, self.pos_y)
        self.arr1[self.pos_x][self.pos_y] = 8
        self.arr2[self.pos_x][self.pos_y] = reward_max

    def reward(self, x1, y1):
        print("reward", x1, y1)
        reward2 = 1 / (math.sqrt((x1 - self.target_x) ** 2 + (y1 - self.target_y) ** 2))
        reward3 = 1 / reward2
        return reward3

    def search(self):
        for i in range(0, 1):
            rnd1 = random.random()
            rnd1 = 0.8130436593743464
            print("rnd", rnd1)
            if rnd1 < self.epsilon:
                self.move_explore()
            else:
                self.move_predict()
            self.print_board(self.arr1)

    def print_board(self, board):
        for row in board:
            for cell in row:
                if int(cell) == 1:
                    print(Fore.RED + "1 ", end="")
                elif int(cell) == 8:
                    print(Fore.YELLOW + "8 ", end="")
                elif int(cell) == 3:
                    print(Fore.GREEN + "3 ", end="")
                else:
                    print(str(int(cell)) + " ", end="")

                print(Fore.RESET, end="")

            print()


w1 = World()

# print(print(w1.arr1))
# print("\nStart Grid")
# w1.print_board(w1.arr1)


w1.search()  # search with random search

# w1.a_star()  # search with AStar
# print("\nAStar Path")
# w1.print_board(w1.arr1)

print("")
print("Arr2")
print(w1.arr2)
print("")

w1.print_board(w1.arr1)
