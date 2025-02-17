import numpy as np


class Reward:
    @staticmethod
    def reward_chaser_calculation(position_victim, position_chaser):
        x, y = position_victim
        x1, y1 = position_chaser
        if x == x1 and y == y1:
            return 10
        distance = np.sqrt((x - x1) ** 2 + (y - y1) ** 2)
        return 0.01 * -distance

    @staticmethod
    def reward_victim_calculation(position_victim, position_chaser):
        reward = 1
        punish = -100
        x, y = position_victim
        x1, y1 = position_chaser
        if x == x1 and y == y1:
            return punish
        return reward
