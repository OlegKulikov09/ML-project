class Reward:
    @staticmethod
    def reward_chaser_calculation(position_victim, position_chaser):
        x, y = position_victim
        x1, y1 = position_chaser
        if x == x1 and y == y1:
            return 10
        #distance = (x-x1)**2 + (y-y1)**2
        #return 1/distance
        return -0.1

"""   @staticmethod
    def reward_victim_calculation(position_victim, position_chaser):
        x, y = position_victim
        x1, y1 = position_chaser
        if x == x1 and y == y1:
            return -10
        distance = (x-x1)**2 + (y-y1)**2
        return distance*0.1 """