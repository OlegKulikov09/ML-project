import random

import numpy as np

from Game import Game
from Player import Player
from Reward import Reward

import matplotlib.pyplot as plt

class Main:
    def __init__(self):
        self.env = Game(10)
        self.output = [0, 1, 2, 3]  # Действия: вверх, вниз, влево, вправо
        self.q_table = {}
        self.alpha = 0.01  # Скорость обучения
        self.gamma = 0.9  # Дисконтирующий фактор
        self.epsilon = 1.0  # Начальное значение ε
        self.epsilon_decay = 0.995  # Уменьшение ε
        self.epsilon_min = 0.1  # Минимальное значение ε

    def get_state(self, chaser_position, victim_position):
        # Преобразуем положение охотника и жертвы в состояние
        return tuple(chaser_position + victim_position)

    def update_q_table(self, state, action, reward, next_state):
        # Обновляем Q-таблицу
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
            self.env.reset()  # Сброс игрового окружения
            done = False
            state = self.get_state(self.env.chaser.position, self.env.victim.position)
            total_reward = 0  # Суммарная награда за эпизод

            while not done:
                if random.uniform(0, 1) < self.epsilon:
                    action = random.choice(self.output)  # Исследование
                else:
                    if state not in self.q_table:
                        self.q_table[state] = np.zeros(len(self.output))
                    action = np.argmax(self.q_table[state])  # Эксплуатация

                next_position = self.env.chaser.move_player(self.env.chaser.position, action)
                reward = Reward.reward_chaser_calculation(self.env.victim.position, next_position)
                total_reward += reward
                next_state = self.get_state(next_position, self.env.victim.position)

                self.update_q_table(state, action, reward, next_state)

                state = next_state
                self.env.chaser.position = next_position  # Обновляем позицию охотника
                #self.env.victim.position = self.env.victim.random_move(self.env.victim.position) # New random victim position

                if self.env.chaser.position == self.env.victim.position:  # Если поймали жертву
                    done = True

            # Уменьшение ε
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            rewards_per_episode.append(total_reward)

        print("Обучение завершено. Q-таблица:")
        print(self.q_table)

        # Построение графика
        plt.plot(rewards_per_episode)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Learning Progress")
        plt.show()

if __name__ == "__main__":
    trainer = Main()
    trainer.train(episodes=300)  # Запускаем обучение с визуализацией