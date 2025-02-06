import random

import numpy as np
import torch
import pandas as pd
from torch import optim, nn

from DQN import DQN
from Game import Game
from PrioritizedReplayBuffer import PrioritizedReplayBuffer
from Reward import Reward

import matplotlib.pyplot as plt


class Main:
    def __init__(self):
        self.env = Game(30)
        self.output = [0, 1, 2, 3]
        self.alpha = 0.0001  # learning rate
        self.gamma = 0.99
        self.epsilon = 1.0  # ε
        self.epsilon_decay = 0.999  # decay of ε
        self.epsilon_min = 0.001  # min value of ε
        self.batch_size = 64
        self.update_target_every = 10
        self.replay_buffer = PrioritizedReplayBuffer(capacity=10000, alpha=0.6)

        # DQN initialization
        input_size = len(self.get_state([0, 0], [0, 0]))
        output_size = len(self.output)
        self.policy_net = DQN(input_size, output_size)
        self.target_net = DQN(input_size, output_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.alpha)
        self.loss_fn = nn.SmoothL1Loss()

    def soft_update(self, tau=0.005):
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)

    def get_state(self, chaser_position, victim_position):
        return tuple(chaser_position + victim_position)

    def store_experience(self, state, action, reward, next_state, done, error):
        self.replay_buffer.add((state, action, reward, next_state, done), error)

    def save_model(self, filename="dqn_model.pth"):
        torch.save(self.policy_net.state_dict(), filename)
        print(f"Model saved to {filename}")

    def load_model(self, filename="dqn_model.pth"):
        self.policy_net.load_state_dict(torch.load(filename))
        self.policy_net.eval()  # Setting test mode
        print(f"Model loaded from {filename}")

    def train_dqn(self):
        if len(self.replay_buffer.buffer) < self.batch_size:
            return

        batch, indices = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        q_values = self.policy_net(states).gather(1, actions)

        # Double DQN
        with torch.no_grad():
            next_actions = torch.argmax(self.policy_net(next_states), dim=1).unsqueeze(1)
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        errors = torch.abs(q_values - target_q_values).detach().numpy().squeeze()
        self.replay_buffer.update_priorities(indices, errors)

        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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
                    with torch.no_grad():
                        state_tensor = torch.tensor([state], dtype=torch.float32)
                        action = torch.argmax(self.policy_net(state_tensor)).item()

                self.env.victim.position = self.env.victim.random_move(self.env.victim.position)
                reward = Reward.reward_chaser_calculation(self.env.victim.position, self.env.chaser.position)
                total_reward += reward

                if self.env.chaser.position == self.env.victim.position or t == self.env.turns:
                    done = True
                    break

                next_position = self.env.chaser.move_player(self.env.chaser.position, action)
                self.env.chaser.position = next_position

                reward = Reward.reward_chaser_calculation(self.env.victim.position, self.env.chaser.position)
                total_reward += reward

                next_state = self.get_state(next_position, self.env.victim.position)

                done = self.env.chaser.position == self.env.victim.position or t == self.env.turns

                with torch.no_grad():
                    state_tensor = torch.tensor([state], dtype=torch.float32)
                    next_state_tensor = torch.tensor([next_state], dtype=torch.float32)
                    q_value = self.policy_net(state_tensor)[0, action]
                    max_next_q_value = torch.max(self.target_net(next_state_tensor)).item()
                    td_error = abs(reward + (1 - done) * self.gamma * max_next_q_value - q_value)

                self.store_experience(state, action, reward, next_state, done, td_error)
                self.train_dqn()

                state = next_state
                t += 1

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            self.soft_update()
            rewards_per_episode.append(total_reward)

        trainer.save_model("trained_model.pth")
        self.plot_rewards(rewards_per_episode)

    def plot_rewards(self, rewards_per_episode):
        window_size = 140
        smoothed_rewards = pd.Series(rewards_per_episode).rolling(window=window_size).mean()
        plt.plot(rewards_per_episode)
        plt.plot(smoothed_rewards, label=f"Smoothed (window={window_size})", color="orange")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Learning Progress")
        plt.show()

    def test(self, episodes):
        successes = 0
        total_reward = 0

        for episode in range(episodes):
            self.env.reset()
            state = self.get_state(self.env.chaser.position, self.env.victim.position)
            done = False
            t = 0

            while not done:
                with torch.no_grad():
                    state_tensor = torch.tensor([state], dtype=torch.float32)
                    action = torch.argmax(self.policy_net(state_tensor)).item()

                #Victim moving logic
                self.env.victim.position = self.env.victim.random_move(self.env.victim.position)
                #Visualisation
                self.env.visualize(self.env.chaser.position, self.env.victim.position)
                reward = Reward.reward_chaser_calculation(self.env.victim.position, self.env.chaser.position)
                total_reward += reward

                if self.env.chaser.position == self.env.victim.position or t == self.env.turns:
                    done = True
                    break

                #Chaser moving logic
                next_position = self.env.chaser.move_player(self.env.chaser.position, action)
                self.env.chaser.position = next_position
                reward = Reward.reward_chaser_calculation(self.env.victim.position, self.env.chaser.position)
                total_reward += reward
                state = self.get_state(self.env.chaser.position, self.env.victim.position)
                #Visualisation2
                self.env.visualize(self.env.chaser.position, self.env.victim.position)

                done = self.env.chaser.position == self.env.victim.position or t == self.env.turns
                t += 1

            self.env.save_gif(f"simulation_episode_{episode}.gif")

            if self.env.chaser.position == self.env.victim.position:
                successes += 1

        accuracy = successes / episodes * 100
        avg_reward = total_reward / episodes

        print(f"Test Results: {successes}/{episodes} successful episodes")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Average Reward: {avg_reward:.2f}")


if __name__ == "__main__":
    trainer = Main()
    #trainer.train(episodes=4000)  # Run train
    trainer.load_model("trained_model.pth") # Test of trained model
    trainer.test(episodes=1)
