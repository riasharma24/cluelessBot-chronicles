import gym

# Choose a Gym environment and specify the render mode
env = gym.make("MountainCarContinuous-v0", render_mode="human")

# Initialize environment
observation = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # Sample a random action
    
    # Use the new API format
    observation, reward, done, truncated, info = env.step(action)

    print(_," ",observation," ",reward," ",done," ",truncated," ",info)
    
    if done or truncated:  # Handle both termination and truncation
        observation = env.reset()  # Reset if the environment is done or truncated

env.close()  # Close the environment after use
