import random

import pygame

from Grid import Grid


class Player:

    def __init__(self, x, y):
        self.position = [x, y]

    @staticmethod
    def draw_visibility(screen, position, color):
        px, py = position
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if abs(dx) + abs(dy) <= 2:
                    nx, ny = px + dx, py + dy
                    if 0 < nx < Grid.GRID_SIZE - 1 and 0 < ny < Grid.GRID_SIZE - 1:
                        rect = pygame.Rect(
                            nx * Grid.CELL_SIZE,
                            ny * Grid.CELL_SIZE,
                            Grid.CELL_SIZE,
                            Grid.CELL_SIZE)
                        pygame.draw.rect(screen, color, rect)

    @staticmethod
    def move_player(position, direction):
        x, y = position
        if direction == "UP" and y > 1: y -= 1
        if direction == "DOWN" and y < Grid.GRID_SIZE - 2: y += 1
        if direction == "LEFT" and x > 1: x -= 1
        if direction == "RIGHT" and x < Grid.GRID_SIZE - 2: x += 1
        return [x, y]

    def random_move(self, position):
        directions = ["UP", "DOWN", "LEFT", "RIGHT"]
        direction = random.choice(directions)
        return self.move_player(position, direction)

    @staticmethod
    def draw_player(screen, x, y, color):
        pygame.draw.circle(
            screen,
            color,
            (x * Grid.CELL_SIZE + Grid.CELL_SIZE // 2, y * Grid.CELL_SIZE + Grid.CELL_SIZE // 2),
            Grid.CELL_SIZE // 3)

class Chaser(Player):
    pass

class Victim(Player):
    pass