# Balancing Double Pendulum with Deep Reinforcement Learning
In thiy Notebook i implemented a double pendulum environment simulation environment suitable for interacting with a deep reinforcement learning agent.
The algorithm used to solve the balancing problem is the Soft Actor Critic Algorithm from:
- [SAC Paper 1](https://arxiv.org/abs/1801.01290)
  provides the original and first version of SAC.
- [SAC Paper 2](https://arxiv.org/abs/1812.05905)
  provides the State-of-the-Art implementation of SAC with automatic temperature parameter optimization.
  
  ## Problem Formulation
  The goal of the RL agent is to balance the two poles in their unstable equilibrium position with the horizontal force on the cart as input control variable.

![alt text](https://www.researchgate.net/profile/Alexander_Bogdanov6/publication/250107215/figure/fig1/AS:669527859798030@1536639289962/Double-inverted-pendulum-on-a-cart.png)

Image Source: https://www.researchgate.net/figure/Double-inverted-pendulum-on-a-cart_fig1_250107215

## Results
The soft actor critic algorithm is able to solve the continuous control task.
In the following image, the learning curve, which is the total return of one episode over all the episodes, is presented:
