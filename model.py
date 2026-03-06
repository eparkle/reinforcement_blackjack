import gymnasium as gym
import random as rand
import matplotlib.pyplot as plt
from collections import defaultdict

env = gym.make('Blackjack-v1', sab=True)

# play a hand based on q table
def play_q_hand(current_state):
    global wins, losses, draws
    action = 0 if q_dict[current_state][0] > q_dict[current_state][1] else 1
    next_state, reward, terminated, truncated, info = env.step(action)
    #base case
    if (terminated or truncated):
        # target = r + y · Q(s', a') = r
        # error = target - Q(s, a) = r - Q(s, a)
        # Q(s, a) = Q(s, a) + lr * error = Q(s, a) + lr * (r - Q(s, a))
        q_dict[current_state][action] += learning_rate * (reward - q_dict[current_state][action])
        if reward == 1:
            wins += 1
        elif reward == -1:
            losses += 1
        else:
            draws += 1
        return q_dict[current_state][action]
    else:
        # target = r + y · Q(s', a') = Q(s', a')
        # error = target - Q(s, a) = Q(s', a') - Q(s, a)
        # Q(s, a) = Q(s, a) + lr * error = Q(s, a) + lr * (Q(s', a') - Q(s, a))
        q_dict[current_state][action] += learning_rate * (play_q_hand(next_state) - q_dict[current_state][action])
        return q_dict[current_state][action]

# play a hand with random actions
def play_epsilon_hand(current_state):
    global wins, losses, draws
    action = rand.randint(0, 1)
    next_state, reward, terminated, truncated, info = env.step(action)
    #base case
    if (terminated or truncated):
        # target = r + y · Q(s', a') = r
        # error = target - Q(s, a) = r - Q(s, a)
        # Q(s, a) = Q(s, a) + lr * error = Q(s, a) + lr * (r - Q(s, a))
        q_dict[current_state][action] += learning_rate * (reward - q_dict[current_state][action])
        if reward == 1:
            wins += 1
        elif reward == -1:
            losses += 1
        else:
            draws += 1
        return q_dict[current_state][action]
    else:
        # target = r + y · Q(s', a') = Q(s', a')
        # error = target - Q(s, a) = Q(s', a') - Q(s, a)
        # Q(s, a) = Q(s, a) + lr * error = Q(s, a) + lr * (Q(s', a') - Q(s, a))
        q_dict[current_state][action] += learning_rate * (play_epsilon_hand(next_state) - q_dict[current_state][action])
        return q_dict[current_state][action]    

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
epsilon_min = 0.05
epochs = 1000000
epsilon_update = 0.9995

#results
wins = 0
losses = 0
draws = 0

#table
q_dict = defaultdict(lambda: [0.0, 0.0]) # keys: (psum, dsum, ace), Values: [q_hit, q_stand]
def evaluate_random_agent(num_games=100000):
    wins = 0
    losses = 0
    draws = 0

    for _ in range(num_games):
        state, info = env.reset()

        while True:
            action = rand.randint(0,1)
            state, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                if reward == 1:
                    wins += 1
                elif reward == -1:
                    losses += 1
                else:
                    draws += 1
                break

    return wins, losses, draws
print(f"Training with learning rate: {learning_rate}, epsilon: {epsilon} for {epochs} iterations")
win_rates = []
iterations = []
interval = 10000   # how often to record performance
for i in range(epochs):
    if i % 200 == 0:
        print(f"Iteration: {i}")
    # get state
    state, info = env.reset()
    player_sum, dealer_sum, usable_ace = state[0], state[1], state[2]
    # if state not seen before initialize with zeros
    if (player_sum, dealer_sum, usable_ace) not in q_dict:
        q_dict[(player_sum, dealer_sum, usable_ace)] = [0, 0]
    # determine if explore and play the corresponding hand
    explore = rand.uniform(0.0, 1.0)
    if explore < epsilon:
        play_epsilon_hand(state)  # explore
    else:
        play_q_hand(state) # exploit
    if epsilon > epsilon_min:
        epsilon *= epsilon_update

    if i % interval == 0 and i > 0:
        total_games = wins + losses + draws
        win_rate = wins / total_games
        win_rates.append(win_rate)
        iterations.append(i)



# Evaluate lower limit
rand_wins, rand_losses, rand_draws = evaluate_random_agent(100000)
random_win_rate = rand_wins / (rand_wins + rand_losses)
plt.plot(iterations, win_rates, label="Trained AI")
plt.axhline(y=random_win_rate, linestyle='--', label="Random Baseline")

plt.xlabel("Training Iterations")
plt.ylabel("Win Rate")
plt.title("Blackjack AI vs Lower Limit")
plt.legend()
plt.show()
