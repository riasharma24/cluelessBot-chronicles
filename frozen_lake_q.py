import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def run(episodes, is_training,render=False):
    env = gym.make('FrozenLake-v1', map_name='8x8', is_slippery=True, render_mode='human' if render else None)

    if(is_training):
       q=np.zeros((env.observation_space.n,env.action_space.n))
    else:
       f=open('frozen_lake8x8.pkl','rb')
       q=pickle.load(f)
       f.close()

    q = np.zeros((env.observation_space.n, env.action_space.n))

    learning_rate_a = 0.1
    discount_factor_g = 0.99

    epsilon = 1
    epsilon_decay_rate = 0.000005
    rng = np.random.default_rng()

    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False
        total_reward = 0

        # Run one episode
        while not (terminated or truncated):
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(q[state, :])  # Exploit best known action

            # Take action and observe the new state and reward
            new_state, reward, terminated, truncated, _ = env.step(action)

            # Update Q-table using Q-learning formula
            if is_training:
                q[state, action] = q[state, action] + learning_rate_a * (reward + discount_factor_g * np.max(q[new_state, :]) - q[state, action])

            state = new_state  # Transition to the new state
            total_reward += reward  # Accumulate reward for this episode

        # Decay epsilon after each episode
        epsilon = max(epsilon - epsilon_decay_rate, 0)

        # Store reward of this episode
        rewards_per_episode[i] = total_reward

        # If exploration is finished, reduce learning rate
        if epsilon == 0:
            learning_rate_a = 0.0001

        if i % 1000 == 0:
            print(f"Episode {i}: Epsilon: {epsilon:.4f}, Total Reward: {total_reward}")


    env.close()

    # Compute the sum of rewards over sliding windows of 100 episodes
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t - 100):t + 1])

    # Plotting the learning curve
    plt.plot(sum_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward (Last 100 Episodes)')
    plt.title('Learning Progress - Frozen Lake 8x8')
    plt.savefig('frozen_lake8x8.png')

    if is_training:
         f=open('frozen_lake8x8.pkl','wb')
         pickle.dump(q, f)
         f.close()

if __name__ == '__main__':
    run(1,False,True)
