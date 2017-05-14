
## Playing Flappy Bird Using Deep Reinforcement Learning (Based on Deep Q Learning DQN)

## Include NIPS 2013 version and Nature Version DQN


I rewrite the code from another repo and make it much simpler and easier to understand Deep Q Network Algorithm from DeepMind

The code of DQN is only 160 lines long.

To run the code, just type python FlappyBirdDQN.py

Since the DQN code is a unique class, you can use it to play other games.


## About the code

As a reinforcement learning problem, we knows we need to obtain observations and output actions, and the 'brain' do the processing work.

Therefore, you can easily understand the BrainDQN.py code. There are three interfaces:

1. getInitState() for initialization
2. getAction()
3. setPerception(nextObservation,action,reward,terminal)

the game interface just need to be able to feed the action to the game and output observation,reward,terminal


## Disclaimer
This work is based on the repo: [yenchenlin1994/DeepLearningFlappyBird](https://github.com/yenchenlin1994/DeepLearningFlappyBird.git)

