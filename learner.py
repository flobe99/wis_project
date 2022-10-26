import math
import random
import numpy as np


class World():
	def __init__(self):
		self.epsilon = 0.2
		self.dim_x = 12
		self.dim_y = 12

		self.arr1 = np.zeros([12, 12])
		self.arr2 = np.zeros([12, 12])

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

		# arr1[target_x][target_y] = 5

	def move_predict(self):
		reward_max = 0
		direction_max = []
		print('predicting, current pos', self.pos_x, self.pos_y)
		for direction in self.directions:
			ftr_x = self.pos_x + direction[0]
			ftr_y = self.pos_y + direction[1]
			if ftr_x >= 0 and ftr_y >= 0 and ftr_x < self.dim_x and ftr_y < self.dim_y:
				print('exploring', direction)
				if self.arr1[ftr_x][ftr_y] == 0:
					x1 = ftr_x
					y1 = ftr_y
					r1 = 1/(math.sqrt((x1-self.target_x)**2 + (y1-self.target_y)**2))
					print('reward', r1)
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
		pass
		# reward_max = 0
		# direction_max = []
		# rnd_directions = []
		# print('exploring, current pos', self.pos_x, self.pos_y)
		# for i1 in range(0, len(directions)):#direction in self.directions:
		# 	direction = self.directions[i1]
		# 	ftr_x = self.pos_x + direction[0]
		# 	ftr_y = self.pos_y + direction[1]
		# 	if ftr_x >= 0 and ftr_y >= 0 and ftr_x < self.dim_x and ftr_y < self.dim_y:
		# 		print('exploring', direction)
		# 		rnd_directions.append(i1)

		# rnd2 = random.randint(0, len(rnd_directions))
		# direction_max = self.directions[rnd_directions[rnd2]]
		# print('direction chosen', direction_max)
		# print('')
		# self.pos_x = self.pos_x + direction_max[0]
		# self.pos_y = self.pos_y + direction_max[1]
		# self.arr1[self.pos_x][self.pos_y] = 8
		# self.arr2[ftr_x][ftr_y] = reward_max

	def reward(x1, y1):
		print('reward', x1, y1)
		reward2 = 1/(math.sqrt((x1-target_x)**2 + (y1-target_y)**2))
		reward3 = 1/reward2
		return reward3

	def search(self):
		rnd1 = random.random()
		# print('rnd', rnd1)
		if rnd1 < self.epsilon:
			self.move_explore()
		else:
			self.move_predict()


w1 = World()
for i in range(0, 10):
	w1.search()
print('')
print(w1.arr2)
print('')
print(w1.arr1)