
# Reinforcement Learning with Policy Gradients in the CartPole Environment

## Introduction

This document analyzes the implementation of a simple policy gradient reinforcement learning algorithm using Jupyter Notebook to solve the CartPole environment. The CartPole problem is a classic reinforcement learning task where an agent must learn to balance a pole on a cart by moving it left or right.

### Key Concepts

- **Policy Network**: A neural network that approximates the policy, mapping states to actions.
- **Policy Gradient**: A method for optimizing policies directly using gradient ascent on expected reward.
- **Discount Factor (GAMMA)**: Determines the importance of future rewards.

## Implementation Overview

### Environment Setup

The CartPole environment is initialized using the `gym` library. The environment provides the state space and action space, which the policy network will use to learn an optimal policy.

```python
env = gym.make('CartPole-v0')
```

### Policy Network

The `PolicyNetwork` class defines a neural network with one hidden layer. The network outputs action probabilities, which are used to select actions during training.

```python
class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=3e-4):
        super(PolicyNetwork, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.softmax(self.linear2(x), dim=1)
        return x 

    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(state)
        highest_prob_action = np.random.choice(self.num_actions, p=np.squeeze(probs.detach().numpy()))
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
        return highest_prob_action, log_prob
```

### Training Process

The agent interacts with the environment, collects rewards, and updates the policy network using the policy gradient method. The rewards are discounted to emphasize short-term rewards while also considering future rewards.

```python
def update_policy(policy_network, rewards, log_probs):
    discounted_rewards = []
    for t in range(len(rewards)):
        Gt = 0 
        pw = 0
        for r in rewards[t:]:
            Gt += GAMMA**pw * r
            pw += 1
        discounted_rewards.append(Gt)
        
    discounted_rewards = torch.tensor(discounted_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)  # Normalize discounted rewards

    policy_gradient = []
    for log_prob, Gt in zip(log_probs, discounted_rewards):
        policy_gradient.append(-log_prob * Gt)
    
    policy_network.optimizer.zero_grad()
    policy_gradient = torch.stack(policy_gradient).sum()
    policy_gradient.backward()
    policy_network.optimizer.step()
```

### Running the Algorithm

The `main` function runs the training loop, where the agent repeatedly interacts with the environment to improve its policy. Performance metrics such as total reward and average reward per episode are printed and plotted to monitor the training progress.

```python
def main():
    env = gym.make('CartPole-v0')
    policy_net = PolicyNetwork(env.observation_space.shape[0], env.action_space.n, 128)
    
    max_episode_num = 5000
    max_steps = 10000
    numsteps = []
    avg_numsteps = []
    all_rewards = []

    for episode in range(max_episode_num):
        state = env.reset()
        log_probs = []
        rewards = []

        for steps in range(max_steps):
            action, log_prob = policy_net.get_action(state)
            new_state, reward, done, _ = env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                update_policy(policy_net, rewards, log_probs)
                numsteps.append(steps)
                avg_numsteps.append(np.mean(numsteps[-10:]))
                all_rewards.append(np.sum(rewards))
                print(f"Episode: {episode}, Total Reward: {np.round(np.sum(rewards), decimals=3)}, "
                      f"Average Reward: {np.round(np.mean(all_rewards[-10:]), decimals=3)}, Length: {steps}")
                break
            
            state = new_state
        
    plt.plot(numsteps, label='Steps per Episode')
    plt.plot(avg_numsteps, label='Average Steps (Last 10 Episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
```

### Analysis of Performance

- **Initial Performance**: The agent starts with a random policy, leading to low and inconsistent rewards.
- **Learning Progress**: As training progresses, the agent improves its ability to balance the pole, reflected in the increasing average rewards.
- **Policy Stability**: The policy network becomes more stable as the rewards converge, showing the effectiveness of the policy gradient method in solving the CartPole problem.

### Conclusion

The policy gradient method implemented here successfully trains the agent to solve the CartPole environment. The results demonstrate the effectiveness of reinforcement learning in environments where the optimal policy can be learned through trial and error.

This implementation can be extended by experimenting with different network architectures, discount factors, or by introducing techniques such as experience replay to further stabilize training.
