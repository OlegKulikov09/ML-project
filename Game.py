import sys
import pygame
import random

from Grid import Grid
from Player import Chaser, Victim
from Reward import Reward


class Game:
    def __init__(self, turns):
        self.turns = turns
        pygame.init()
        self.screen = pygame.display.set_mode((Grid.SCREEN_SIZE, Grid.SCREEN_SIZE))
        pygame.display.set_caption("NN Tags")
        self.clock = pygame.time.Clock()
        self.size = Grid.GRID_SIZE

        self.chaser = Chaser(2, 2)
        self.victim = Victim(Grid.GRID_SIZE - 3, Grid.GRID_SIZE - 3)
        self.running = True

    def visualize(self, chaser_action, victim_action):
        self.screen.fill(Grid.WHITE)
        Grid.draw_grid(self.screen)

        if chaser_action is not None:
            self.chaser.position = self.chaser.move_player(self.chaser.position, chaser_action)
        if victim_action is not None:
            self.victim.position = self.victim.move_player(self.victim.position, victim_action)

        self.chaser.draw_player(self.screen, *self.chaser.position, Grid.BLUE)
        self.victim.draw_player(self.screen, *self.victim.position, Grid.RED)

        pygame.display.flip()
        self.clock.tick(2)  # FPS

        # Check game status
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def reset(self):
        self.chaser = Chaser(2, 2)
        self.victim = Victim(Grid.GRID_SIZE - 3, Grid.GRID_SIZE - 3)
        self.running = True