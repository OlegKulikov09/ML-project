
import pygame

class Grid:
    CELL_SIZE = 50
    GRID_SIZE = 10
    SCREEN_SIZE = CELL_SIZE * GRID_SIZE
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    BLUE = (0, 0, 255)
    RED = (255, 0, 0)
    GRAY = (200, 200, 200)
    LIGHT_BLUE = (173, 216, 230)
    LIGHT_RED = (255, 182, 193)

    @staticmethod
    def draw_grid(screen):
        for x in range(Grid.GRID_SIZE):
            for y in range(Grid.GRID_SIZE):
                rect = pygame.Rect(x * Grid.CELL_SIZE, y * Grid.CELL_SIZE, Grid.CELL_SIZE, Grid.CELL_SIZE)
                color = Grid.GRAY if (
                        x <= 0 or
                        y <= 0 or
                        x == Grid.GRID_SIZE - 1 or
                        y == Grid.GRID_SIZE - 1 or
                        x == Grid.GRID_SIZE - 1 or
                        y == Grid.GRID_SIZE - 1) \
                    else Grid.WHITE
                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, Grid.BLACK, rect, 1)