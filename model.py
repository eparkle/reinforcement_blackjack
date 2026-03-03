import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm

print(f"Gymnasium version: {gym.__version__}")

# Create the environment
env = gym.make('Blackjack-v1', sab=True)  # sab=True uses Sutton & Barto rules

print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")
print(f"Actions: 0=Stand, 1=Hit")

# Play a few sample episodes manually to understand the environment
print("=== Sample Episodes ===")

for episode in range(5):
    state, info = env.reset()
    print(f"\nEpisode {episode + 1}:")
    print(f"  Initial state: player_sum={state[0]}, dealer_card={state[1]}, usable_ace={state[2]}")
    
    done = False
    step = 0
    while not done:
        # Random action for demonstration
        action = env.action_space.sample()
        action_name = 'Hit' if action == 1 else 'Stand'
        
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1
        
        print(f"  Step {step}: Action={action_name}, "
              f"New state={next_state}, Reward={reward}, Done={done}")
        
        state = next_state
    
    result = 'Win' if reward > 0 else ('Draw' if reward == 0 else 'Lose')
    print(f"  Result: {result}")

    def evaluate_policy(env, policy_fn, n_episodes=100000):
        """
        Evaluate a policy over many episodes.
        
        Args:
            env: Gymnasium environment
            policy_fn: Function that takes a state and returns an action
            n_episodes: Number of episodes to simulate
        
        Returns:
            win_rate, draw_rate, lose_rate
        """
        wins, draws, losses = 0, 0, 0
        
        for _ in range(n_episodes):
            state, _ = env.reset()
            done = False
            
            while not done:
                action = policy_fn(state)
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            
            if reward > 0:
                wins += 1
            elif reward == 0:
                draws += 1
            else:
                losses += 1
        
        return wins / n_episodes, draws / n_episodes, losses / n_episodes


# Random policy
random_policy = lambda state: env.action_space.sample()
win_rate, draw_rate, lose_rate = evaluate_policy(env, random_policy)
print(f"Random Policy: Win={win_rate:.3%}, Draw={draw_rate:.3%}, Lose={lose_rate:.3%}")

# Simple threshold policy (stand on 17+)
threshold_policy = lambda state: 1 if state[0] < 17 else 0
win_rate, draw_rate, lose_rate = evaluate_policy(env, threshold_policy)
print(f"Threshold (17) Policy: Win={win_rate:.3%}, Draw={draw_rate:.3%}, Lose={lose_rate:.3%}")



"""
Actions:
    Hit
    Stand
Vars:
    r: reward
    y: discount factor
    s: current state
    a: current action
    s': future state
    a': future action
    lr: learning rate
    e: epsilon
Formulas:
    Target = r + y · max Q(s', a') --> target = r * max Q(s', a')
        For blackjack scenario y = 1
    Error = target - Q(s, a)
    Q(s, a) = Q(s, a) + lr * error
"""

#hyperparams
learning_rate = 0.01
epsilon = 1.0
epochs = 1000
epsilon_update = epsilon / epochs

#results
wins = 0
losses = 0
draws = 0

#table
q_dict = {} # keys: (psum, dsum, ace), Values: [q_hit, q_stand]

print(f"Training with learning rate: {learning_rate}, epsilon: {epsilon} for {epochs} iterations")

for i in epochs:
    if i % 50 == 0:
        print(f"Iteration: {i}")
    # get state
    state, info = env.reset()
    player_sum, dealer_sum, usable_ace = state[0], state[1], state[2]
    # if state not seen before initialize with zeros
    if (player_sum, dealer_sum, usable_ace) not in q_dict:
        q_dict[(player_sum, dealer_sum, usable_ace)] = [0, 0]

    

    play_hand()

    # recursive function that plays the hand
    def play_hand(current_state):
        action =  'Hit' if q_dict[current_state][0] > q_dict[current_state][1] else 'Stand'
        next_state, reward, terminated, truncated, info = env.step(action)
        if (terminated or truncated):
            
            return reward
        else:
            return play_hand(next_state)


