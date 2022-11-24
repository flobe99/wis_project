"""
Author(s):  AA228 Group 100
Date:       Nov 15, 2019
Desc:       Obstacle
"""

import numpy as np

class Obstacle():
    def __init__(self, pos, penalty):
        self.pos = np.array(pos, dtype=int)
        self.penalty = penalty