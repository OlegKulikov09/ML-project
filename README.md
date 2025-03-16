# ABOUT THE PROJECT

This project was developed as part of the Machine Learning course in the Robotics and AI program at Alpen-Adria University of Klagenfurt. The goal was to apply Deep Reinforcement Learning techniques to a predator-prey scenario. The project was evaluated with the highest grade (1.0), demonstrating strong algorithmic design and practical implementation of DQN-based agents.

During development, the following components were created and described: environment, two agents, training processes, and execution of trained agents. The PyTorch library was used as the training framework. For a visually appealing episode display, the pygame library was used.

## Project Goal
The goal of the project is to simulate a game of tag, where one agent acts as a hunter searching for prey, while the other learns to evade the hunter. The agents do not have any predefined directives, algorithms, or behavior scripts; their decisions must be based solely on experience gained through reinforcement learning.

## Key Technologies
	•	Deep Reinforcement Learning (DQN)
	•	PyTorch (training framework)
	•	Prioritized Experience Replay (improving training efficiency)
	•	Pygame (visualization)
	•	Python (main programming language)

# PROGRAM STRUCTURE

- **Game.py**: Contains functions for initializing the environment and visualizing episodes.
- **Player.py**: Describes methods for initializing agents and their functions (moving in different directions, random movement for training purposes).
- **Grid.py**: Manages the game grid, including rendering and updating the environment.
- **Reward.py**: Defines the reward functions for reinforcement learning. The hunter agent is rewarded for capturing the prey and penalized for excessive distance. The prey agent is rewarded for maintaining distance and penalized when caught.
- **DQN.py**: Defines the neural network initialization. It consists of two hidden layers with 128 neurons each. The input data includes agent position coordinates, and the output determines the next step. The activation function used is the classic ReLU.
- **PrioritizedReplayBuffer.py**: Ensures the correct training process by retrieving not just the most recent episodes (standard approach) but also random episodes from storage. This provides the agent with more variability and efficiency in exploration and decision-making.
- **Main.py**: The `Main` class is the core of the reinforcement learning framework. It initializes the environment, sets up Deep Q-Networks (DQN) for training agents, and provides methods for training and testing both the hunter and victim agents.

#### Key Components:
- **Initialization (`__init__`)**: Sets up the game environment, hyperparameters, DQN networks, optimizer, loss function, and replay buffer.
- **State Representation (`get_state`)**: Converts game positions into a format suitable for the neural network.
- **Experience Storage (`store_experience`)**: Adds experiences to the prioritized replay buffer.
- **Model Saving/Loading (`save_model`, `load_model`)**: Saves and loads the trained models.
- **Training (`train_dqn`, `train`, `train_victim`)**:
  - `train_dqn()`: Updates the DQN using prioritized experience replay.
  - `train()`: Trains the hunter agent by interacting with the environment.
  - `train_victim()`: Trains the victim agent against a pre-trained hunter model.
- **Testing (`test`)**: Runs a test simulation where a trained hunter and victim agent interact.
- **Plotting (`plot_rewards`)**: Visualizes the training progress over episodes.

  The main script that executes training and testing procedures for the agents:
  
  - Starts the training process for the hunter agent, with the number of training episodes specified as a parameter:
    ```python
    trainer.train(2000)
    ```
  - Starts the training process for the prey agent, specifying the number of episodes and the hunter model used for training:
    ```python
    trainer.train_victim(episodes=4000, hunter_model_path="trained_model.pth")
    ```
  - Runs an episode with trained agents:
    ```python
    trainer.test(1, "trained_model.pth", "trained_victim.pth")
    ```

The pure development time for the current version took 14 days. Including task formulation, strategy research, refactoring, and fine-tuning, the total duration was approximately two months.

An extended version with a more complex environment can be found in a colleague’s repository: [Hunter-Prey-DQN](https://github.com/GrzegorzHimself/Hunter-Prey-DQN)

# TRAINING MATERIALS

- **Hunter agent training graph:**  
  ![Training Graph](https://github.com/OlegKulikov09/ML-project/blob/main/Figure_1.png)

- **Animation of a trained hunter and prey agent with random movement:**  
  ![Hunter Trained](https://github.com/OlegKulikov09/ML-project/blob/main/Hunter_trained.gif)

- **Prey agent training graph:**  
  ![Prey Training](https://github.com/OlegKulikov09/ML-project/blob/main/prey%20train%202%204k.png)

- **Animation of a trained prey agent and hunter agent:**  
  ![Prey Trained](https://github.com/OlegKulikov09/ML-project/blob/main/Prey_trained.gif)

# FUTURE DEVELOPMENT

Potential improvements include:

1. **Environment complexity:** Adding obstacles or interactive objects to the field. Introducing a feature for randomly generating the game field for each new episode.
2. **Advanced Agent Perception:** Replace coordinate-based inputs with vision-based data (e.g., convolutional neural networks) to enhance decision-making capabilities.

# 📩 CONTACT
If you have any questions or suggestions, feel free to reach out: olkulikov@edu.aau.at.
