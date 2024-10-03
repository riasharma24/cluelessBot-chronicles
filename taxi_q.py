import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def run(episodes, is_training=True, render=False):
    env = gym.make('Taxi-v3', render_mode='human' if render else None)

    if is_training:
        q = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        with open('taxi.pkl', 'rb') as f:
            q = pickle.load(f)

    learning_rate_a = 0.05  # Lower learning rate for stable learning
    discount_factor_g = 0.99  # Higher discount factor for long-term rewards

    epsilon = 1
    epsilon_decay_rate = 0.0001  # Faster epsilon decay for quicker exploitation
    rng = np.random.default_rng()

    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False
        total_reward = 0

        while not (terminated or truncated):
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(q[state, :])  # Exploit best known action

            new_state, reward, terminated, truncated, _ = env.step(action)

            # If terminal, update Q-value with reward only (no future state)
            if is_training:
                if terminated or truncated:
                    q[state, action] = reward
                else:
                    q[state, action] += learning_rate_a * (
                        reward + discount_factor_g * np.max(q[new_state, :]) - q[state, action]
                    )

            state = new_state
            total_reward += reward

        epsilon = max(epsilon - epsilon_decay_rate, 0)
        rewards_per_episode[i] = total_reward

        if epsilon == 0:
            learning_rate_a = 0.0001

        if i % 1000 == 0:
            print(f"Episode {i}: Epsilon: {epsilon:.4f}, Total Reward: {total_reward}")

    env.close()

    # Sum of rewards over 100 episode windows
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t - 100):t + 1])

    plt.plot(sum_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward (Last 100 Episodes)')
    plt.title('Learning Progress - Taxi-v3')
    plt.savefig('taxi.png')

    if is_training:
        with open('taxi.pkl', 'wb') as f:
            pickle.dump(q, f)

if __name__ == '__main__':
    run(15000)  # Train for 15,000 episodes
    run(10, False, True)  # Test for 10 episodes
