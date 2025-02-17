import numpy as np
import pygame

from Grid import Grid
from Player import Chaser, Victim
from PIL import Image


class Game:
    def __init__(self, turns):
        self.turns = turns
        self.frames = []
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

        self.victim.draw_player(self.screen, *self.victim.position, Grid.BLUE)
        self.chaser.draw_player(self.screen, *self.chaser.position, Grid.RED)

        pygame.display.flip()
        self.clock.tick(2)  # FPS

        frame = pygame.surfarray.array3d(pygame.display.get_surface())
        frame = np.transpose(frame, (1, 0, 2))
        self.frames.append(Image.fromarray(frame))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def reset(self):
        self.chaser = Chaser(2, 2)
        self.victim = Victim(Grid.GRID_SIZE - 3, Grid.GRID_SIZE - 3)
        self.running = True

    def save_gif(self, filename="animation.gif", fps=2):
        if self.frames:
            self.frames[0].save(filename, save_all=True, append_images=self.frames[1:], duration=1000 // fps, loop=0)
            print(f"GIF saved as {filename}")
