import sys

import pygame

from Grid import Grid
from Player import Player, Chaser, Victim


class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((Grid.SCREEN_SIZE, Grid.SCREEN_SIZE))
        pygame.display.set_caption("Chaser and Victim")
        self.clock = pygame.time.Clock()

        # Игроки
        self.chaser = Chaser(1, 1)
        self.victim = Victim(Grid.GRID_SIZE - 2, Grid.GRID_SIZE - 2)
        self.turn = 1
        self.running = True

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

                if event.type == pygame.KEYDOWN:
                    if self.turn == 1:
                        if event.key == pygame.K_w:
                            self.chaser.position = Player.move_player(self.chaser.position, "UP")
                        if event.key == pygame.K_s:
                            self.chaser.position = Player.move_player(self.chaser.position, "DOWN")
                        if event.key == pygame.K_a:
                            self.chaser.position = Player.move_player(self.chaser.position, "LEFT")
                        if event.key == pygame.K_d:
                            self.chaser.position = Player.move_player(self.chaser.position, "RIGHT")
                        self.turn = 2
                    elif self.turn == 2:
                        if event.key == pygame.K_UP:
                            self.victim.position = Player.move_player(self.victim.position, "UP")
                        if event.key == pygame.K_DOWN:
                            self.victim.position = Player.move_player(self.victim.position, "DOWN")
                        if event.key == pygame.K_LEFT:
                            self.victim.position = Player.move_player(self.victim.position, "LEFT")
                        if event.key == pygame.K_RIGHT:
                            self.victim.position = Player.move_player(self.victim.position, "RIGHT")
                        self.turn = 1

            if self.chaser.position == self.victim.position:
                print("Chaser caught the victim!")
                self.running = False

            self.screen.fill(Grid.BLACK)
            Grid.draw_grid(self.screen)
            Player.draw_visibility(self.screen, self.chaser.position, Grid.LIGHT_BLUE)
            Player.draw_visibility(self.screen, self.victim.position, Grid.LIGHT_RED)
            Player.draw_player(self.screen, *self.chaser.position, Grid.BLUE)
            Player.draw_player(self.screen, *self.victim.position, Grid.RED)
            pygame.display.flip()

            self.clock.tick(30)

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = Game()
    game.run()
