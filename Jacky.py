import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

class JackyBase:
    def __init__(self, max_sum=21, epsilon=0.1, gamma=0.99):
        self.max_sum = max_sum
        self.epsilon = epsilon
        self.actions = ["Hit", "Stay"]

    def policy(self, S):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def run_episode(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def train(self, num_episodes=10000):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def evaluate_policy(self, num_episodes=100):
        total_reward = 0
        for _ in range(num_episodes):
            S = 0
            while True:
                action = self.policy(S)
                if action == "Stay":
                    total_reward += S
                    break

                card = random.randint(1, 10)
                S += card
                if S > self.max_sum:
                    break
        
        avg_reward = total_reward / num_episodes
        print(f"Average reward over {num_episodes} episodes: {avg_reward:.2f}")
        return avg_reward


class Jacky(JackyBase):
    def __init__(self, max_sum=21, epsilon=0.1):
        super().__init__(max_sum, epsilon)
        self.Q_table = {s: {action: 0 for action in self.actions} for s in range(max_sum + 1)}
        self.returns = {s: {action: [] for action in self.actions} for s in range(max_sum + 1)}

    def policy(self, S):
        """ε-Greedy policy"""
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        return max(self.Q_table[S], key=self.Q_table[S].get)

    def run_episode(self):
        S = 0
        episode = []
        while True:
            action = self.policy(S)
            episode.append((S, action))

            if action == "Stay":
                reward = S
                break

            card = random.randint(1, 10)
            S += card

            if S > self.max_sum:
                reward = 0
                break

        return episode, reward

    def update_Q(self, episode, reward):
        visited = set()
        for S, action in episode:
            if (S, action) not in visited:
                self.returns[S][action].append(reward)
                self.Q_table[S][action] = sum(self.returns[S][action]) / len(self.returns[S][action])
                visited.add((S, action))

    def train(self, num_episodes=10000):
        for _ in range(num_episodes):
            episode, reward = self.run_episode()
            self.update_Q(episode, reward)

class JackyNN(JackyBase):
    def __init__(self, max_sum=21, epsilon=0.1, gamma=0.99):
        super().__init__(max_sum, epsilon, gamma)
        self.model = self.build_model()

    def build_model(self):
        model = models.Sequential()
        model.add(layers.InputLayer(input_shape=(22 + 21,)))  # 22 for sum (one-hot encoded) + 21 rewards
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1, activation='linear'))  # Output Q-value for "Hit"
        model.compile(optimizer='adam', loss='mse')
        return model

    def policy(self, S, rewards):
        """ε-Greedy policy"""
        state = self.encode_state(S, rewards)
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        q_value = self.model.predict(np.expand_dims(state, axis=0))[0][0]
        return "Hit" if q_value > 0 else "Stay"

    def encode_state(self, S, rewards):
        state = [0] * (self.max_sum + 1)  # 1-hot encoding for S
        if S <= self.max_sum:
            state[S] = 1
        state += rewards
        return np.array(state)

    def run_episode(self):
        S = 0
        rewards = [random.randint(1, 21) for _ in range(21)]  # Generate random rewards at the start of the episode
        episode = []
        while True:
            action = self.policy(S, rewards)
            episode.append((S, action))
            if action == "Stay":
                reward = rewards[S] if 0 <= S < len(rewards) else 0  # Use pre-generated rewards
                break
            card = random.randint(1, 10)
            S += card
            if S > self.max_sum:
                reward = 0
                break
        return episode, reward, rewards

    def update_model(self, episode, reward, rewards):
        for i, (S, action) in enumerate(reversed(episode)):
            if action == "Hit":
                state = self.encode_state(S, rewards)
                target = reward * (self.gamma ** i)  # Discounted reward
                self.model.fit(np.expand_dims(state, axis=0), np.array([target]), verbose=0)

    def train(self, num_episodes=10000):
        for _ in range(num_episodes):
            episode, reward, rewards = self.run_episode()
            self.update_model(episode, reward, rewards)

def main(): 
    game_choice = input("Choose the game to play (1 for Jacky, 2 for JackyNN): ")
    jacky = Jacky() if game_choice == "1" else JackyNN() if game_choice == "2" else None
    if jacky is None: raise ValueError("Invalid choice. Please select 1 or 2.")
    print("Training the model...")
    jacky.train(num_episodes=1000)
    print("Evaluating the policy...")
    jacky.evaluate_policy(num_episodes=100)

if __name__ == "__main__":
    main()