import math
import statistics
import random
import numpy as np
from colorama import Fore, Back, Style
import time
import sys

from astar import AStar
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn import svm
from scipy.spatial import ConvexHull, convex_hull_plot_2d


# https://www.geeksforgeeks.org/ml-monte-carlo-tree-search-mcts/


class World:
    def __init__(self):
        self.epsilon = 0.2
        self.dim_x = 12
        self.dim_y = 12

        self.arr1 = np.zeros([self.dim_x, self.dim_y])
        self.arr2 = np.zeros([self.dim_x, self.dim_y])

        self.pos_x = 0
        self.pos_y = 0

        self.target_x = 11
        self.target_y = 1

        self.reset_board()

        self.directions = [[0, 0], [0, 1], [0, -1], [1, 0], [1, 1], [1, -1], [-1, 0], [-1, 1], [-1, -1]]

        self.possible_directions = []

        self.leaf_id = 1
        self.tree_width = 3
        self.tree_horizon = 1

        self.samples_x = []
        self.samples_y = []

        self.treeleaves = []
        self.treeleaves2 = []
        self.treeleaves2_index = 0

        self.treeleaves.append(
            {"id": self.leaf_id, "parent": 0, "parent_action": [0], "reward": -5, "position": [self.pos_x, self.pos_y]}
        )
        self.leaf_id = self.leaf_id + 1

        self.treeleaves2.append(
            {
                "id": self.treeleaves2_index,
                "parent": 0,
                "mean_x": self.dim_x * 0.5,
                "mean_y": self.dim_y * 0.5,
                "dst": max(self.dim_x, self.dim_y) * 0.5,
                "clf": svm.SVC(kernel="linear"),
            }
        )

    def a_star_cost_array(self):
        temp_arr = np.copy(self.arr1)
        self.arr1[self.arr1 == 1] = 100
        self.arr1[self.arr1 == 0] = 1
        cost_arr = self.arr1
        self.arr1 = np.copy(temp_arr)
        return cost_arr

    def a_star(self):
        start = (self.pos_x, self.pos_y)
        goal = (self.target_x, self.target_y)

        astar = AStar(self.arr1)
        path, sample_count = astar.search(start, goal)

        # find maximum reward in grid
        max_reward = float("-inf")
        for row in self.arr1:
            for col in row:
                max_reward = max(max_reward, col)

        print("Maximum reward:", max_reward)
        print("Sample Count:", sample_count)

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
        print("choosing max pos, current pos", self.pos_x, self.pos_y)
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

        print("direction chosen", direction_max)
        print("")
        self.pos_x = self.pos_x + direction_max[0]
        self.pos_y = self.pos_y + direction_max[1]
        self.arr1[self.pos_x][self.pos_y] = 8
        self.arr2[ftr_x][ftr_y] = reward_max

    def move_explore(self):
        rnd1 = random.randint(0, len(self.possible_directions))
        index_dir = self.possible_directions[rnd1]
        print("chosing rnd pos, current pos", self.pos_x, self.pos_y)
        direction = self.directions[index_dir]
        print("direction chosen", direction)
        ftr_x = self.pos_x + direction[0]
        ftr_y = self.pos_y + direction[1]
        self.pos_x = self.pos_x + direction[0]
        self.pos_y = self.pos_y + direction[1]
        self.arr1[self.pos_x][self.pos_y] = 8
        # self.arr2[ftr_x][ftr_y] = reward_max

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

    def take_samples(self, num_samples):
        for i1 in range(0, num_samples):
            rndx = random.randint(0, self.dim_x - 1)
            rndy = random.randint(0, self.dim_y - 1)
            self.samples_x.append([rndx, rndy])
            self.samples_y.append([self.reward(rndx, rndy)])

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

    def reset_board(self):

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

    def treesearch_simple(self, hm_loops):
        for i in range(0, hm_loops):
            print("treesearch", i)
            self.tree_select(1)

    def tree_select(self, leaf_id):
        print("tree_select", leaf_id)
        leaves_arr = []
        reward_temp = -5
        index_temp = 0

        for leaf1 in self.treeleaves:
            if leaf1["parent"] == leaf_id:
                leaves_arr.append(leaf1["id"])
                if leaf1["reward"] > reward_temp:
                    index_temp = leaf1["id"]
                    reward_temp = leaf1["reward"]
        print("treeleaves", leaves_arr)

        if len(leaves_arr) <= 0:
            print("tr exp")
            self.tree_expand(leaf_id)
        else:
            rnd1 = random.random()
            # rnd1 = 0.8130436593743464
            if rnd1 < self.epsilon:
                rnd2 = random.randint(0, len(leaves_arr) - 1)
                print("rnd sel", rnd2)
                self.tree_select(leaves_arr[rnd2])
            else:
                print("rew sel")
                self.tree_select(index_temp)

    def tree_expand(self, leaf_id):
        print("tree_expand")
        arr1 = self.tree_sim(leaf_id)
        leaves_add = min(len(arr1), self.tree_width)
        print(self.directions)
        print(arr1)
        for i in range(0, leaves_add):
            direction1 = arr1[i][1]
            ftr_x = self.pos_x + self.directions[direction1][0]
            ftr_y = self.pos_y + self.directions[direction1][1]
            if self.arr1[ftr_x][ftr_y] != 2:
                self.treeleaves.append(
                    {
                        "id": self.leaf_id,
                        "parent": leaf_id,
                        "parent_action": [arr1[i][1]],
                        "reward": arr1[i][0],
                        "position": [ftr_x, ftr_y],
                    }
                )
                self.leaf_id = self.leaf_id + 1
                self.arr1[ftr_x][ftr_y] = 2
        # print('treeleaves')
        # for treeleaf in self.treeleaves:
        #     print(treeleaf)
        # print('')

    def tree_sim(self, leaf_id):
        print("tree_sim")
        list1 = []
        for leaf1 in self.treeleaves:
            if leaf1["id"] == leaf_id:
                arr_pos = leaf1["position"]
                self.pos_x = arr_pos[0]
                self.pos_y = arr_pos[1]
        self.get_directions()
        # print('directions poss')
        # print(self.possible_directions)
        selection = random.sample(self.possible_directions, 3)
        # print(self.directions)
        for dir1 in selection:
            direction = self.directions[dir1]
            ftr_x = self.pos_x + direction[0]
            ftr_y = self.pos_y + direction[1]
            reward1 = self.reward(ftr_x, ftr_y)
            list1.append([reward1, dir1])
        list1.sort(reverse=True)
        # print(list1)
        return list1

    def tree_backprop(self, leaf_id):
        pass

    def treesearch_partition(self):
        self.sample()

    def treewalk(self):
        reward_temp = -5
        id_temp = 0
        dir_list = []
        for leaf1 in self.treeleaves:
            if leaf1["reward"] > reward_temp:
                reward_temp = leaf1["reward"]
                id_temp = leaf1["id"]

        while id_temp != 0:
            for leaf2 in self.treeleaves:
                if leaf2["id"] == id_temp:
                    id_temp = leaf2["parent"]
                    dir_list = leaf2["parent_action"] + dir_list
        print("dirlist")
        print(dir_list)
        self.pos_x = self.treeleaves[0]["position"][0]
        self.pos_y = self.treeleaves[0]["position"][1]
        for dir1 in dir_list:
            self.pos_x = self.pos_x + self.directions[dir1][0]
            self.pos_y = self.pos_y + self.directions[dir1][1]
            self.arr1[self.pos_x][self.pos_y] = 8

    def lamcts(self, leaf_id):
        clcx = 0
        clcy = 0
        sampling_dst = 0
        leaf_temp = {}
        for leaf2 in self.treeleaves2:
            if leaf2["id"] == leaf_id:
                clcx = leaf2["mean_x"]
                clcy = leaf2["mean_y"]
                leaf_temp = leaf2
                sampling_dst = leaf2["dst"]
        X1 = []
        y1 = []
        for i1 in range(0, 15):
            rndx = random.randint(int(max(0, clcx - sampling_dst)), int(min(self.dim_x - 1, clcx + sampling_dst)))
            rndy = random.randint(int(max(0, clcy - sampling_dst)), int(min(self.dim_y - 1, clcy + sampling_dst)))
            if self.arr1[rndx][rndy] != 1:
                X1.append([rndx, rndy])
                y1.append([self.reward(rndx, rndy)])
            else:
                i1 = i1 - 1
        print("X1", X1)
        print("y1", y1)
        print("kmeans labels")
        kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(y1)
        labels1 = kmeans.labels_
        print(labels1)
        print("kmeans centers")
        centers1 = kmeans.cluster_centers_
        print(centers1)

        pred_index_good = 5
        pred_index_bad = 5
        if centers1[0] < centers1[1]:
            pred_index_good = 1
            pred_index_bad = 0
        else:
            pred_index_good = 0
            pred_index_bad = 1

        clf = svm.SVC(kernel="linear")
        clf.fit(X1, labels1)
        print("clf2", clf.support_vectors_)
        leaf_temp["clf"] = clf

        lbl_x1 = []
        lbl_x2 = []
        lbl_y1 = []
        lbl_y2 = []
        for i2 in range(0, len(labels1)):
            x2 = X1[i2][0]
            y2 = X1[i2][1]
            print("x:", x2, "y:", y2)
            print("label:", labels1[i2])
            print("")
            if labels1[i2] == pred_index_bad:
                lbl_x1.append(x2)
                lbl_y1.append(y2)
            if labels1[i2] == pred_index_good:
                lbl_x2.append(x2)
                lbl_y2.append(y2)
            # if labels1[i2] == 1:
            #     self.arr1[x2][y2] = 2
            # if labels1[i2] == 0:
            #     self.arr1[x2][y2] = 9

        x_mean1 = round(statistics.mean(lbl_x1), 0)
        y_mean1 = round(statistics.mean(lbl_y1), 0)
        x_mean2 = round(statistics.mean(lbl_x2), 0)
        y_mean2 = round(statistics.mean(lbl_y2), 0)

        distance_mean = math.sqrt((x_mean1 - x_mean2) ** 2 + (y_mean1 - y_mean2) ** 2)

        self.treeleaves2_index = self.treeleaves2_index + 1
        d1 = {
            "id": self.treeleaves2_index,
            "parent": leaf_id,
            "mean_x": x_mean1,
            "mean_y": y_mean1,
            "dst": distance_mean,
            "clf": svm.SVC(kernel="linear"),
        }
        self.treeleaves2_index = self.treeleaves2_index + 1
        d2 = {
            "id": self.treeleaves2_index,
            "parent": leaf_id,
            "mean_x": x_mean2,
            "mean_y": y_mean2,
            "dst": distance_mean,
            "clf": svm.SVC(kernel="linear"),
        }
        self.treeleaves2.append(d1)
        self.treeleaves2.append(d2)
        if self.treeleaves2_index < 3:
            self.lamcts(d1["id"])
            self.lamcts(d2["id"])

        for i3 in range(0, self.dim_x):
            for j3 in range(0, self.dim_y):
                if self.arr1[i3][j3] != 1:
                    pred = clf.predict([[i3, j3]])
                    if pred[0] == 0:
                        self.arr1[i3][j3] = 9
                    if pred[0] == 1:
                        self.arr1[i3][j3] = 2
        # print('prediction', clf.predict([[3, 10]]))

    def plot_lamcts(self):
        plt.figure(1)
        plt.grid()
        plt.plot(
            [-0.5, self.dim_x - 0.5, self.dim_x - 0.5, -0.5, -0.5],
            [0.5, 0.5, self.dim_y * -1 + 0.5, self.dim_y * -1 + 0.5, 0.5],
            color="black",
        )
        for i in range(0, self.dim_x):
            for j in range(0, self.dim_y):
                if self.arr1[i][j] == 1:
                    plt.scatter(j, -i, color="black", marker="s")
        for treeleaf in self.treeleaves:
            plt.scatter(treeleaf["position"][1], treeleaf["position"][0] * -1, color="black", s=100, alpha=0.20)
            for leaf2 in self.treeleaves:
                if leaf2["id"] == treeleaf["parent"]:
                    pos2x = leaf2["position"][0]
                    pos2y = leaf2["position"][1]
                    plt.plot(
                        [treeleaf["position"][1], pos2y],
                        [treeleaf["position"][0] * -1, pos2x * -1],
                        color="blue",
                        linestyle="dashed",
                    )
        for treeleaf2 in self.treeleaves2:
            print(treeleaf2)
            color1 = "black"
            try:
                print(treeleaf2["clf"].support_vectors_)
                color1 = "green"
            except:
                pass
            print("")
            # plt.scatter(treeleaf2['mean_x'], -treeleaf2['mean_y'], color=color1, marker='^')
            # plt.text(treeleaf2['mean_x'], -treeleaf2['mean_y'], treeleaf2['id'])
        clf3 = self.treeleaves2[0]["clf"]
        clf4 = self.treeleaves2[2]["clf"]

        points1 = []
        points2 = []

        for i4 in range(0, self.dim_x):
            for j4 in range(0, self.dim_y):
                if self.arr1[i4][j4] != 1:
                    test4 = clf3.predict([[i4, j4]])
                    test5 = clf4.predict([[i4, j4]])
                    if test4 == 1:
                        # plt.scatter(j4, i4*-1, color='blue', s=100, alpha=0.5)
                        points1.append([i4 * -1, j4])
                    else:
                        # plt.scatter(j4, i4*-1, color='red', s=100, alpha=0.5)
                        points2.append([i4 * -1, j4])

        hull1 = ConvexHull(points1)
        hull2 = ConvexHull(points2)

        for simplex in hull1.simplices:
            plt.plot(np.array(points1)[simplex, 1], np.array(points1)[simplex, 0], color="blue", linestyle="dotted")
        for simplex2 in hull2.simplices:
            plt.plot(np.array(points2)[simplex2, 1], np.array(points2)[simplex2, 0], color="red", linestyle="dotted")

        plt.scatter(self.treeleaves2[1]["mean_x"], self.treeleaves2[1]["mean_y"] * -1)
        plt.scatter(self.treeleaves2[2]["mean_x"], self.treeleaves2[2]["mean_y"] * -1)

        plt.show()

    def lap3(self, leaf_id, samples_xvec, samplex_yvec):
        clcx = 0
        clcy = 0
        sampling_dst = 0
        leaf_temp = {}
        for leaf2 in self.treeleaves2:
            if leaf2["id"] == leaf_id:
                clcx = leaf2["mean_x"]
                clcy = leaf2["mean_y"]
                leaf_temp = leaf2
                sampling_dst = leaf2["dst"]
        X1 = []
        y1 = []
        for i1 in range(0, 15):
            rndx = random.randint(int(max(0, clcx - sampling_dst)), int(min(self.dim_x - 1, clcx + sampling_dst)))
            rndy = random.randint(int(max(0, clcy - sampling_dst)), int(min(self.dim_y - 1, clcy + sampling_dst)))
            if self.arr1[rndx][rndy] != 1:
                self.samples_x.append([rndx, rndy])
                self.samples_y.append([self.reward(rndx, rndy)])
                X1.append([rndx, rndy])
                y1.append([self.reward(rndx, rndy)])
            else:
                i1 = i1 - 1
        print("X1", X1)
        print("y1", y1)
        print("kmeans labels")
        kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(y1)
        labels1 = kmeans.labels_
        print(labels1)
        print("kmeans centers")
        centers1 = kmeans.cluster_centers_
        print(centers1)

        pred_index_good = 5
        pred_index_bad = 5
        if centers1[0] < centers1[1]:
            pred_index_good = 1
            pred_index_bad = 0
        else:
            pred_index_good = 0
            pred_index_bad = 1

        clf = svm.SVC(kernel="linear")
        clf.fit(X1, labels1)
        print("clf2", clf.support_vectors_)
        leaf_temp["clf"] = clf

        lbl_x1 = []
        lbl_x2 = []
        lbl_y1 = []
        lbl_y2 = []
        for i2 in range(0, len(labels1)):
            x2 = X1[i2][0]
            y2 = X1[i2][1]
            print("x:", x2, "y:", y2)
            print("label:", labels1[i2])
            print("")
            if labels1[i2] == pred_index_bad:
                lbl_x1.append(x2)
                lbl_y1.append(y2)
            if labels1[i2] == pred_index_good:
                lbl_x2.append(x2)
                lbl_y2.append(y2)
            # if labels1[i2] == 1:
            #     self.arr1[x2][y2] = 2
            # if labels1[i2] == 0:
            #     self.arr1[x2][y2] = 9

        x_mean1 = round(statistics.mean(lbl_x1), 0)
        y_mean1 = round(statistics.mean(lbl_y1), 0)
        x_mean2 = round(statistics.mean(lbl_x2), 0)
        y_mean2 = round(statistics.mean(lbl_y2), 0)

        distance_mean = math.sqrt((x_mean1 - x_mean2) ** 2 + (y_mean1 - y_mean2) ** 2)

        self.treeleaves2_index = self.treeleaves2_index + 1
        d1 = {
            "id": self.treeleaves2_index,
            "parent": leaf_id,
            "mean_x": x_mean1,
            "mean_y": y_mean1,
            "dst": distance_mean,
            "clf": svm.SVC(kernel="linear"),
        }
        self.treeleaves2_index = self.treeleaves2_index + 1
        d2 = {
            "id": self.treeleaves2_index,
            "parent": leaf_id,
            "mean_x": x_mean2,
            "mean_y": y_mean2,
            "dst": distance_mean,
            "clf": svm.SVC(kernel="linear"),
        }
        self.treeleaves2.append(d1)
        self.treeleaves2.append(d2)
        if self.treeleaves2_index < 3:
            self.lamcts(d1["id"])
            self.lamcts(d2["id"])

        for i3 in range(0, self.dim_x):
            for j3 in range(0, self.dim_y):
                if self.arr1[i3][j3] != 1:
                    pred = clf.predict([[i3, j3]])
                    if pred[0] == 0:
                        self.arr1[i3][j3] = 9
                    if pred[0] == 1:
                        self.arr1[i3][j3] = 2
        # print('prediction', clf.predict([[3, 10]]))

    def plot_lap3(self):
        plt.figure(1)
        plt.grid()
        plt.plot(
            [-0.5, self.dim_x - 0.5, self.dim_x - 0.5, -0.5, -0.5],
            [0.5, 0.5, self.dim_y * -1 + 0.5, self.dim_y * -1 + 0.5, 0.5],
            color="black",
        )
        for i in range(0, self.dim_x):
            for j in range(0, self.dim_y):
                if self.arr1[i][j] == 1:
                    plt.scatter(j, -i, color="black", marker="s")
        for treeleaf in self.treeleaves:
            plt.scatter(treeleaf["position"][1], treeleaf["position"][0] * -1, color="black", s=100, alpha=0.20)
            for leaf2 in self.treeleaves:
                if leaf2["id"] == treeleaf["parent"]:
                    pos2x = leaf2["position"][0]
                    pos2y = leaf2["position"][1]
                    plt.plot(
                        [treeleaf["position"][1], pos2y],
                        [treeleaf["position"][0] * -1, pos2x * -1],
                        color="blue",
                        linestyle="dashed",
                    )
        for treeleaf2 in self.treeleaves2:
            print(treeleaf2)
            color1 = "black"
            try:
                print(treeleaf2["clf"].support_vectors_)
                color1 = "green"
            except:
                pass
            print("")
            # plt.scatter(treeleaf2['mean_x'], -treeleaf2['mean_y'], color=color1, marker='^')
            # plt.text(treeleaf2['mean_x'], -treeleaf2['mean_y'], treeleaf2['id'])
        clf3 = self.treeleaves2[0]["clf"]
        clf4 = self.treeleaves2[2]["clf"]

        points1 = []
        points2 = []

        for i4 in range(0, self.dim_x):
            for j4 in range(0, self.dim_y):
                if w1.arr1[i4][j4] != 1:
                    test4 = clf3.predict([[i4, j4]])
                    test5 = clf4.predict([[i4, j4]])
                    if test4 == 1:
                        # plt.scatter(j4, i4*-1, color='blue', s=100, alpha=0.5)
                        points1.append([i4 * -1, j4])
                    else:
                        # plt.scatter(j4, i4*-1, color='red', s=100, alpha=0.5)
                        points2.append([i4 * -1, j4])

        hull1 = ConvexHull(points1)
        hull2 = ConvexHull(points2)

        for simplex in hull1.simplices:
            plt.plot(np.array(points1)[simplex, 1], np.array(points1)[simplex, 0], color="blue", linestyle="dotted")
        for simplex2 in hull2.simplices:
            plt.plot(np.array(points2)[simplex2, 1], np.array(points2)[simplex2, 0], color="red", linestyle="dotted")

        plt.scatter(self.treeleaves2[1]["mean_x"], self.treeleaves2[1]["mean_y"] * -1)
        plt.scatter(self.treeleaves2[2]["mean_x"], self.treeleaves2[2]["mean_y"] * -1)

        plt.show()


def execute_astar():
    print("execute astar")
    w1_astar = World()
    w1_astar.a_star()  # search with AStar
    print("\nAStar Path")
    w1_astar.print_board(w1_astar.arr1)


def execute_mcts():
    print("execute mcts")
    # print(print(w1_mcts.arr1))
    # print("\nStart Grid")
    # w1_mcts.print_board(w1_mcts.arr1)

    # w1_mcts.treesearch_simple(800)
    w1_mcts = World()
    w1_mcts.take_samples(15)
    w1_mcts.lamcts(0)

    print("board1")
    w1_mcts.print_board(w1_mcts.arr1)
    w1_mcts.treewalk()  # search with random search

    print("")
    print("Arr2")
    print(w1_mcts.arr2)
    print("")

    w1_mcts.print_board(w1_mcts.arr1)
    w1_mcts.plot_lamcts()


def execute_lap3():
    w1_lap3 = World()


def main():
    execute_astar()
    execute_mcts()
    execute_lap3()


if __name__ == "__main__":
    main()
