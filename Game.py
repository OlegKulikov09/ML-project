import sys
import pygame
import random

from Grid import Grid
from Player import Chaser, Victim
from Reward import Reward


class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((Grid.SCREEN_SIZE, Grid.SCREEN_SIZE))
        pygame.display.set_caption("NN Tags")
        self.clock = pygame.time.Clock()

        self.chaser = Chaser(1, 1)
        self.victim = Victim(Grid.GRID_SIZE - 2, Grid.GRID_SIZE - 2)
        self.turns = 6
        self.running = True

        self.chaser_rewards = []
        self.victim_rewards = []

    def run(self):
        round_number = 1

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            if round_number > self.turns:
                self.running = False
                continue

            print(f"Round {round_number}:")

            self.chaser.position = self.chaser.random_move(self.chaser.position)
            chaser_reward = Reward.reward_chaser_calculation(self.victim.position, self.chaser.position)
            self.chaser_rewards.append(chaser_reward)

            if self.chaser.position == self.victim.position:
                print(f"Chaser caught the victim!")
                chaser_reward = Reward.reward_chaser_calculation(self.victim.position, self.chaser.position)
                victim_reward = Reward.reward_victim_calculation(self.victim.position, self.chaser.position)

                self.chaser_rewards[-1] = chaser_reward
                self.victim_rewards.append(victim_reward)
                self.draw_game_state()
                break

            self.victim.position = self.victim.random_move(self.victim.position)
            victim_reward = Reward.reward_victim_calculation(self.victim.position, self.chaser.position)
            self.victim_rewards.append(victim_reward)

            self.draw_game_state()

            round_number += 1

            self.clock.tick(4)

        self.show_statistics()

        pygame.quit()
        sys.exit()

    def draw_game_state(self):
        self.screen.fill(Grid.BLACK)
        Grid.draw_grid(self.screen)
        Chaser.draw_visibility(self.screen, self.chaser.position, Grid.LIGHT_BLUE)
        Victim.draw_visibility(self.screen, self.victim.position, Grid.LIGHT_RED)
        Chaser.draw_player(self.screen, *self.chaser.position, Grid.BLUE)
        Victim.draw_player(self.screen, *self.victim.position, Grid.RED)
        pygame.display.flip()

    def show_statistics(self):
        total_chaser_reward = sum(self.chaser_rewards)
        total_victim_reward = sum(self.victim_rewards)

        print("\nGame Over!")
        print(f"Chaser Rewards: {self.chaser_rewards}")
        print(f"Victim Rewards: {self.victim_rewards}")
        print(f"Total Chaser Reward: {total_chaser_reward}")
        print(f"Total Victim Reward: {total_victim_reward}")


if __name__ == "__main__":
    game = Game()
    game.run()