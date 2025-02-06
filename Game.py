import pygame
import imageio

from Grid import Grid
from Player import Chaser, Victim
from PIL import Image
import numpy as np


class Game:
    def __init__(self, turns):
        self.frames = []
        self.turns = turns
        pygame.init()
        self.screen = pygame.display.set_mode((Grid.SCREEN_SIZE, Grid.SCREEN_SIZE))
        pygame.display.set_caption("NN Tags")
        self.clock = pygame.time.Clock()
        self.size = Grid.GRID_SIZE

        self.chaser = Chaser(1, 1)
        self.victim = Victim(Grid.GRID_SIZE - 2, Grid.GRID_SIZE - 2)
        self.running = True

    def visualize(self, chaser_action, victim_action):
        self.screen.fill(Grid.WHITE)
        Grid.draw_grid(self.screen)

        if victim_action is not None:
            self.victim.position = self.victim.move_player(self.victim.position, victim_action)
        if chaser_action is not None:
            self.chaser.position = self.chaser.move_player(self.chaser.position, chaser_action)

        self.victim.draw_player(self.screen, *self.victim.position, Grid.BLUE)
        self.chaser.draw_player(self.screen, *self.chaser.position, Grid.RED)

        pygame.display.flip()
        self.clock.tick(4)  # FPS

        frame = pygame.surfarray.array3d(pygame.display.get_surface())  # Получаем RGB-изображение
        frame = np.transpose(frame, (1, 0, 2))  # Меняем оси (Pygame хранит их в другом порядке)
        self.frames.append(Image.fromarray(frame))  # Добавляем кадр в список

        # Check game status
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def reset(self):
        self.chaser = Chaser(1, 1)
        self.victim = Victim(Grid.GRID_SIZE - 2, Grid.GRID_SIZE - 2)
        self.running = True

    def save_gif(self, filename="animation.gif", fps=2):
        if self.frames:
            self.frames[0].save(filename, save_all=True, append_images=self.frames[1:], duration=1000//fps, loop=0)
            print(f"GIF сохранён как {filename}")