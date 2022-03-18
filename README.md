# Keras implementation of Branching Dueling Q-Network (BDQ algorithm) 

This is an implementation of a Keras version of the Branching Dueling Deep Q-Learning algorithm. It is based on https://github.com/MoMe36/BranchingDQN, on the paper https://arxiv.org/pdf/1711.08946.pdf and their implementation https://github.com/atavakol/action-branching-agents/tree/master/agents/bdq

BDQ allows a Q-Learning agent to select multiple actions simultaneously, it scales linearly with the action space dimension, thus solving the 'curse of dimentionality' problem for the DQN algorithm. The same principle could also be used for other RL algorithms that suffer from the curse of action space dimensionality... 

This BDQ implementation in Keras is demonstrated on `BipedalWalker-v3` environment.


## How to use: 

To train an agent, run: 

```bash
python train.py
```

To see the agent perform: 
```bash
python enjoy.py
```

## Performances

![BipedalWalker-perf](./runs/BipedalWalker-v3_tf/reward.png)
