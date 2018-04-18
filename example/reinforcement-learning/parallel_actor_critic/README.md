# 'Parallel Advantage-Actor Critic' Implementation

This repo contains a MXNet implementation of a variant of the A3C algorithm from [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/pdf/1602.01783v2.pdf).

Trajectories are obtained from multiple environments in a single process, batched together, and used to update the model with a single forward and backward pass.

[Generalized Advantage Estimation](https://arxiv.org/pdf/1506.02438v5.pdf) is used to estimate the advantage function.

Please see the accompanying [tutorial](https://minpy.readthedocs.io/en/latest/tutorial/rl_policy_gradient_tutorial/rl_policy_gradient.html#improved-rl-with-parallel-advantage-actor-critic) for additional background.

Author: Sean Welleck ([@wellecks](https://github.com/wellecks)), Reed Lee ([@loofahcus](https://github.com/loofahcus))


## Prerequisites
  - Install Scikit-learn: `python -m pip install --user sklearn`
  - Install SciPy: `python -m pip install --user scipy`
  - Install the required OpenAI environments. For example, install Atari: `pip install gym[atari]`

For more details refer: https://github.com/openai/gym

## Training

#### Atari Pong

The model can be trained on various OpenAI gym environments, but was primarily tested on `PongDeterministic-v3`. To train on
this environment with default parameters (16 environments), use:

```bash
python train.py
```

Training a model to achieve a score of 20 takes roughly an hour on a Macbook Pro.

#### Other environments

Note that other environments may require additional tuning or architecture adjustments. Use `python train.py -h` to see the command-line arguments.
For instance, to train on `CartPole-v0`, performing updates every 50 steps,
use:

```bash
python train.py --env-type CartPole-v0 --t-max 50
```
