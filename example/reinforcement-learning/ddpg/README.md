# mx-DDPG
MXNet Implementation of DDPG

# Introduction

This is the MXNet implementation of [DDPG](https://arxiv.org/abs/1509.02971). It is tested in the rllab cart pole environment against rllab's native implementation and achieves comparably similar results. You can substitute with this anywhere you use rllab's DDPG with minor modifications.

# Dependency

* rllab

# Usage

To run the algorithm, 

```python
python run.py
```

The implementation relies on rllab for environments and logging and the hyperparameters could be set in ```run.py```.

# References

1. [Lillicrap, Timothy P., et al. Continuous control with deep reinforcement learning. arXiv preprint arXiv:1509.02971 (2015)](https://arxiv.org/abs/1509.02971).
2. rllab. URL: [https://github.com/openai/rllab](https://github.com/openai/rllab)
