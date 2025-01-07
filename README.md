### README for Jacky Game Implementation

This repository contains two different implementations of the **Jacky** game using two distinct approaches: **Q-Learning** and **Deep Q-Learning**. Below is an explanation of each class and how to use the program.

---

## Overview

### Game Description
The **Jacky** game is a simplified version of a card game where the goal is to accumulate points by drawing cards. Each card drawn increases the player's score. The player can either **Hit** (draw another card) or **Stay** (end the game and keep their current score). The game ends when the player exceeds the maximum score (21), or when the player chooses to stay.

### Models Used

1. **Jacky (Q-Learning)**:
   - A traditional **Q-Learning** agent that uses a Q-table to store and update action values for each state (score).
   - The agent chooses actions based on the **ε-Greedy** policy.
   - It updates its Q-values using the **Monte Carlo method** with returns from completed episodes.
   
2. **JackyNN (Deep Q-Learning)**:
   - A **Deep Q-Learning** agent that uses a neural network to approximate the Q-values.
   - The neural network takes the current score and a set of rewards as input and predicts the Q-value for the action **Hit**.
   - The agent updates its model using the **backpropagation** method after each episode.

---

## How to Use

1. **Installation**:
   Make sure you have the following libraries installed:
   ```bash
   pip install numpy tensorflow
   ```

2. **Running the Program**:
   The program will ask you to choose which game to play. You can choose either:

   - **1**: To play the traditional **Jacky** game (Q-Learning).
   - **2**: To play the **JackyNN** game (Deep Q-Learning).

   The program will then train the chosen model and evaluate its performance.

---

## Code Walkthrough

### Classes

1. **`JackyBase`**:
   - A base class that contains common functionality for both the Q-Learning and Deep Q-Learning models, including:
     - `evaluate_policy`: Evaluates the model by playing a number of episodes and calculating the average reward.

2. **`Jacky`**:
   - Inherits from `JackyBase` and implements the **Q-Learning** algorithm:
     - `policy`: Implements the ε-Greedy policy for action selection.
     - `run_episode`: Simulates a game episode.
     - `update_Q`: Updates the Q-table using the Monte Carlo method.

3. **`JackyNN`**:
   - Inherits from `JackyBase` and implements the **Deep Q-Learning** algorithm using a neural network:
     - `build_model`: Builds the neural network for approximating Q-values.
     - `policy`: Selects actions based on the output of the neural network.
     - `update_model`: Updates the neural network using backpropagation.

### Main Program Flow

1. **User Input**:
   The program prompts the user to choose between two game modes:
   - **1**: Run the Q-Learning version of the game (`Jacky`).
   - **2**: Run the Deep Q-Learning version of the game (`JackyNN`).
   - If the user enters an invalid choice, the program raises an error.

2. **Training and Evaluation**:
   After selecting the game, the program will:
   - Train the model using the `train()` method.
   - Evaluate the model's performance using the `evaluate_policy()` method and print the average reward.

---

## Example of Running the Program

```
Choose the game to play (1 for Jacky, 2 for JackyNN): 1
Training the model...
Evaluating the policy...
Average reward over 100 episodes: 15.61
```

OR

```
Choose the game to play (1 for Jacky, 2 for JackyNN): 2
Training the model...
Evaluating the policy...
Average reward over 100 episodes: 10.68
```

### Performance Results

- **Jacky (Q-Learning)**: Achieved an average reward of **15.61** over 100 episodes.
- **JackyNN (Deep Q-Learning)**: Achieved an average reward of **10.68** over 100 episodes.

This demonstrates that the Q-Learning model (Jacky) performed better in this specific case than the Deep Q-Learning model (JackyNN).

---

## Conclusion

- The **Jacky (Q-Learning)** model performs better in terms of average reward in this implementation, which may be due to the simplicity of the task that doesn't require the complexity of a neural network.
- **JackyNN (Deep Q-Learning)**, while more complex, might perform better in more complex environments or with different reward structures.

Both models can be further optimized, and adjustments can be made to improve performance, such as tuning the learning rate, exploration rate, or even changing the architecture of the neural network.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
