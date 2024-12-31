import random

import numpy as np
import torch
import pandas as pd

from Game import Game
from Player import Player
from Reward import Reward

import matplotlib.pyplot as plt


class Main:
    def __init__(self):
        self.env = Game(30)
        self.output = [0, 1, 2, 3]
        self.q_table = {}
        self.alpha = 0.01  # learning rate
        self.gamma = 0.9
        self.epsilon = 1.0  # ε
        self.epsilon_decay = 0.95  # decay of ε
        self.epsilon_min = 0.1  # min value of ε

    def get_state(self, chaser_position, victim_position):
        return tuple(chaser_position + victim_position)

    def update_q_table(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.output))
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(len(self.output))

        self.q_table[state][action] += self.alpha * (
                reward + self.gamma * np.max(self.q_table[next_state]) - self.q_table[state][action]
        )

    def train(self, episodes):
        rewards_per_episode = []
        for episode in range(episodes):
            self.env.reset()
            done = False
            state = self.get_state(self.env.chaser.position, self.env.victim.position)
            total_reward = 0
            t = 0

            while not done:
                if random.uniform(0, 1) < self.epsilon:
                    action = random.choice(self.output)  # exploration
                else:
                    if state not in self.q_table:
                        self.q_table[state] = np.zeros(len(self.output))
                    action = np.argmax(self.q_table[state])  # exploitation

                self.env.victim.position = self.env.victim.random_move(
                    self.env.victim.position)  # New random victim position
                next_position = self.env.chaser.move_player(self.env.chaser.position, action)
                reward = Reward.reward_chaser_calculation(self.env.victim.position, next_position)
                total_reward += reward
                next_state = self.get_state(next_position, self.env.victim.position)

                self.update_q_table(state, action, reward, next_state)

                state = next_state
                self.env.chaser.position = next_position  # new chaser position
                t += 1

                #if episode == 1 or episode == 50 or episode == 300 or episode == 1000 or episode == 2000:
                 #   self.env.visualize(self.env.chaser.position, self.env.victim.position)

                if self.env.chaser.position == self.env.victim.position or t == self.env.turns:  # end game condition
                    done = True

            # recalculating ε
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            rewards_per_episode.append(total_reward)

        print("Train finished. Q-table:")
        print(self.q_table)

        # Graph drawing
        window_size = 80
        smoothed_rewards = pd.Series(rewards_per_episode).rolling(window=window_size).mean()
        plt.plot(rewards_per_episode)
        plt.plot(smoothed_rewards, label=f"Smoothed (window={window_size})", color="orange")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Learning Progress")
        plt.show()


if __name__ == "__main__":
    trainer = Main()
    trainer.train(episodes=3000)  # Run
