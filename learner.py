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

        self.possible_directions = []

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

        self.leaf_id = 1
        self.tree_width = 3
        self.tree_horizon = 1

        self.treeleaves = []

        self.treeleaves.append({'id':self.leaf_id, 'parent':0, 'parent_action': [0], 'reward': -5, 'position':[self.pos_x, self.pos_y]})
        self.leaf_id = self.leaf_id + 1

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

    def get_directions(self):
        self.possible_directions = []
        # print('predicting, current pos', self.pos_x, self.pos_y)
        for i1 in range(0, len(self.directions)):
            direction = self.directions[i1]
            ftr_x = self.pos_x + direction[0]
            ftr_y = self.pos_y + direction[1]
            if ftr_x >= 0 and ftr_y >= 0 and ftr_x < self.dim_x and ftr_y < self.dim_y:
                if self.arr1[ftr_x][ftr_y] != 1:
                    # print('possible direction', self.directions[i1])
                    self.possible_directions.append(i1)

    def move_predict(self):
        reward_max = 0
        direction_max = []
        print('choosing max pos, current pos', self.pos_x, self.pos_y)
        for i1 in self.possible_directions:
            direction = self.directions[i1]
            ftr_x = self.pos_x + direction[0]
            ftr_y = self.pos_y + direction[1]
            x1 = ftr_x
            y1 = ftr_y
            r1 = self.reward(x1, y1)
            # r1 = 1/(math.sqrt((x1-self.target_x)**2 + (y1-self.target_y)**2))
            if r1 > reward_max:
                reward_max = r1
                direction_max = direction

        print('direction chosen', direction_max)
        print('')
        self.pos_x = self.pos_x + direction_max[0]
        self.pos_y = self.pos_y + direction_max[1]
        self.arr1[self.pos_x][self.pos_y] = 8
        self.arr2[ftr_x][ftr_y] = reward_max

    def move_explore(self):
        rnd1 = random.randint(0, len(self.possible_directions))
        index_dir = self.possible_directions[rnd1]
        print('chosing rnd pos, current pos', self.pos_x, self.pos_y)
        direction = self.directions[index_dir]
        print('direction chosen', direction)
        ftr_x = self.pos_x + direction[0]
        ftr_y = self.pos_y + direction[1]
        self.pos_x = self.pos_x + direction[0]
        self.pos_y = self.pos_y + direction[1]
        self.arr1[self.pos_x][self.pos_y] = 8
        # self.arr2[ftr_x][ftr_y] = reward_max

    def reward(self, x1, y1):
        # print('reward', x1, y1)
        reward2 = 1 / (math.sqrt((x1 - self.target_x) ** 2 + (y1 - self.target_y) ** 2))
        # reward3 = 1 / reward2
        # print(reward2, reward3)
        return reward2

    def search(self):
        for i in range(0, 1):
            rnd1 = random.random()
            rnd1 = 0.8130436593743464
            print("rnd", rnd1)
            self.get_directions()
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


    def treewalk(self):
        reward_temp = -5
        id_temp = 0
        dir_list = []
        for leaf1 in self.treeleaves:
            if leaf1['reward'] > reward_temp:
                reward_temp = leaf1['reward']
                id_temp = leaf1['id']

        while(id_temp != 0):
            for leaf2 in self.treeleaves:
                if leaf2['id'] == id_temp:
                    id_temp = leaf2['parent']
                    dir_list.append(leaf2['parent_action'][0])
        print('dirlist')
        print(dir_list)
        self.pos_x = self.treeleaves[0]['position'][0]
        self.pos_y = self.treeleaves[0]['position'][1]
        for dir1 in dir_list:
            self.pos_x = self.pos_x + self.directions[dir1][0]
            self.pos_y = self.pos_y + self.directions[dir1][1]
            self.arr1[self.pos_x][self.pos_y] = 8

    def treesearch(self):
        for i in range(0, 20):
            print('treesearch', i)
            self.tree_select(1)


    def tree_select(self, leaf_id):
        print('tree_select', leaf_id)
        leaves_arr = []
        reward_temp = -5
        index_temp = 0

        for leaf1 in self.treeleaves:
            if leaf1['parent'] == leaf_id:
                leaves_arr.append(leaf1['id'])
                if leaf1['reward'] > reward_temp:
                    index_temp = leaf1['id']
                    reward_temp = leaf1['reward']

        if len(leaves_arr) <= 0:
            self.tree_expand(leaf_id)
        else:
            rnd1 = random.random()
            # rnd1 = 0.8130436593743464
            if rnd1 < self.epsilon:
                rnd2 = random.randint(0, len(leaves_arr))
                self.tree_select(rnd2)
            else:
                self.tree_select(index_temp)

    #self.treeleaves.append({'id':self.leaf_id, 'parent':0, 'partent_action': [0], 'reward': -5, 'position':[self.pos_x, self.pos_y]})

    def tree_expand(self, leaf_id):
        print('tree_expand')
        arr1 = self.tree_sim(leaf_id)
        leaves_add = min(len(arr1), self.tree_width)
        print(self.directions)
        for i in range(0, leaves_add):
            direction1 = arr1[i][1]
            ftr_x = self.pos_x + self.directions[direction1][0]
            ftr_y = self.pos_y + self.directions[direction1][1]
            self.treeleaves.append({'id':self.leaf_id, 'parent':leaf_id, 'parent_action': [arr1[i][1]], 'reward': arr1[i][0], 'position':[ftr_x, ftr_y]})
            self.leaf_id = self.leaf_id + 1
        print('treeleaves')
        print(self.treeleaves)
        print('')


    def tree_sim(self, leaf_id):
        print('tree_sim')
        list1 = []
        for leaf1 in self.treeleaves:
            if leaf1['id'] == leaf_id:
                arr_pos = leaf1['position']
                self.pos_x = arr_pos[0]
                self.pos_y = arr_pos[1]
        self.get_directions()
        # print(self.possible_directions)
        # print(self.directions)
        for dir1 in self.possible_directions:
            direction = self.directions[dir1]
            ftr_x = self.pos_x + direction[0]
            ftr_y = self.pos_y + direction[1]
            reward1 = self.reward(ftr_x, ftr_y)
            list1.append([reward1, dir1])
        list1.sort(reverse=True)
        print(list1)
        return list1

    def tree_backprop(self, leaf_id):
        pass



w1 = World()

# print(print(w1.arr1))
# print("\nStart Grid")
# w1.print_board(w1.arr1)

w1.treesearch()


w1.treewalk()  # search with random search

# w1.a_star()  # search with AStar
# print("\nAStar Path")
w1.print_board(w1.arr1)

print("")
print("Arr2")
print(w1.arr2)
print("")

w1.print_board(w1.arr1)

# print(w1.treeleaves)
