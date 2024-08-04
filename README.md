# Taxi Navigation with Reinforcement Learning

This project implements reinforcement learning algorithms to solve a taxi navigation problem using the OpenAI Gym's Taxi-v3 environment. The project aims to apply Q-learning and SARSA algorithms to teach a taxi agent how to navigate a grid world efficiently, picking up and dropping off passengers at designated locations with minimal steps.

## Table of Contents

- Introduction
- Problem Context
- Environment Setup
- Action Selection Methods
- Reinforcement Learning Algorithms
- Training the Models
- Evaluation
- Results and Discussion

## Introduction

In this project, we explore the application of reinforcement learning techniques to navigate a taxi in a grid world. The task is to control a taxi to pick up passengers and drop them off at specific locations, optimizing the route to minimize the number of steps taken.

## Problem Context

The problem is modeled using the Taxi-v3 environment from OpenAI Gym. The environment consists of a 5x5 grid with four designated locations marked by R(ed), G(reen), Y(ellow), and B(lue). The taxi must navigate this grid to complete its task:

- Start from a random position.
- Pick up a passenger located at one of the designated spots.
- Drive to the passenger's destination.
- Drop off the passenger.

The goal is to train the taxi to perform these tasks efficiently using reinforcement learning methods. The environment provides three rendering modes: "human," "rgb_array," and "ansi," with the "ansi" mode used for text-based visualizations.

### Actions

The taxi can perform six discrete actions:

1. **Move South** (Action 0)
2. **Move North** (Action 1)
3. **Move East** (Action 2)
4. **Move West** (Action 3)
5. **Pickup Passenger** (Action 4)
6. **Drop off Passenger** (Action 5)

The reward structure is as follows:

- **+20** for successfully dropping off a passenger.
- **-10** for illegal pickup or drop-off actions.
- **-1** for each time step taken.

### Objective

The main objective is for the agent (taxi) to learn to navigate the grid world and complete its task with the minimum number of steps possible. This involves fine-tuning hyperparameters such as the learning rate (\(\alpha\)), exploration parameters (ϵ or T), and discount factor (\(\gamma\)) to achieve optimal performance.

## Environment Setup

To set up the environment, follow these steps:

1. **Install the required packages:**

   ```bash
   pip install gym
   pip install pygame
   ```

2. **Initialize the Taxi-v3 environment:**

   ```python
   import gym
   env = gym.make('Taxi-v3', render_mode='ansi')
   state = env.reset()
   rendered_env = env.render()
   print(rendered_env)
   ```

3. **Understand the environment:**

   - **State Space**: The environment has \(500\) states, each representing a unique configuration of the taxi and passenger.
   - **Action Space**: There are \(6\) actions as described earlier.

## Action Selection Methods

### greedy / ϵ-greedy

  The $\epsilon$-greedy strategy is a commonly used algorithm in reinforcement learning to balance exploration and exploitation. It selects a random action with probability $\epsilon$ and selects the action with the highest Q-value with probability \(1 - $\epsilon$).
  
  1. With probability $\epsilon$, select a random action:
     $`\
     a_t = \text{random action}
     `$
  
  2. With probability \(1 - $\epsilon$), select the action that maximizes the Q-value:
     $`\
     a_t = \arg\max_{a} Q(s_t, a)
     `$
### softmax
  The Softmax method is a more refined exploration and exploitation balancing strategy. In this method, the probability of each action being selected is proportional to the exponential value of its estimated value. The probability $P(a)$ of selecting action $a$ at time $t$ is calculated using the following formula:

  $`\
  P(a) = \frac{\exp(Q_t(a) / \tau)}{\sum_{a'} \exp(Q_t(a') / \tau)}
  `$
  
  where $\tau$ is the temperature parameter that controls the balance between exploration and exploitation:
  
  - When $\tau$ is large (high temperature), the probabilities of selecting all actions become closer to equal, resulting in more exploration.
  - When $\tau$ is small (low temperature), the probability of selecting actions with higher estimated values is significantly higher, resulting in more exploitation.

## Reinforcement Learning Algorithms

This project implements two reinforcement learning algorithms:

### Q-learning

Q-learning is an off-policy algorithm that aims to learn the optimal action-value function to make the best action decision for each state. By continually updating the Q-values, the Q-learning algorithm progressively optimizes the policy, enabling the selection of actions that maximize the long-term cumulative reward in any given state.

### SARSA (State-Action-Reward-State-Action)

SARSA is an on-policy algorithm that updates the action-value function based on the current action taken and the next action chosen by the policy.

## Evaluation

After training, the models are evaluated on 100 random test episodes using the greedy action selection method. The performance metrics include:

- **Average Accumulated Reward**: The average reward obtained per episode.
- **Average Steps per Episode**: The mean number of steps taken to complete each episode.

### Q-learning Evaluation

- **Target**: Perform at most 14 steps per episode and obtain a minimum of 7 average accumulated rewards.

### SARSA Evaluation

- **Target**: Perform at most 15 steps per episode and obtain a minimum of 5 average accumulated rewards.

## Results and Discussion

The results are plotted to show the learning progress and performance:

- **Accumulated Reward per Episode**: Visualizes the total reward accumulated over episodes.
- **Steps per Episode**: Displays the number of steps taken by the agent in each episode.

These plots help analyze the effectiveness of different exploration parameters and learning rates.

### Visualizing the Agent

The trained agent is visualized in the Taxi-v3 environment, showing its decision-making process and reward accumulation over time.

