import numpy as np


class BaseStrategy(object):
    """
    Base class of exploration strategy.
    """

    def get_action(self, obs, policy):

        raise NotImplementedError

    def reset(self):

        pass


class OUStrategy(BaseStrategy):
    """
    Ornstein-Uhlenbeck process: dxt = theta * (mu - xt) * dt + sigma * dWt
    where Wt denotes the Wiener process.
    """

    def __init__(self, env_spec, mu=0, theta=0.15, sigma=0.3):

        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.action_space = env_spec.action_space
        self.state = np.ones(self.action_space.flat_dim) * self.mu
        
    def evolve_state(self):

        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx

        return self.state

    def reset(self):

        self.state = np.ones(self.action_space.flat_dim) * self.mu

    def get_action(self, obs, policy):

        # get_action accepts a 2D tensor with one row
    	obs = obs.reshape((1, -1))
        action = policy.get_action(obs)
        increment = self.evolve_state()
        
        return np.clip(action + increment, 
                       self.action_space.low, 
                       self.action_space.high)


if __name__ == "__main__":

    class Env1(object):

        def __init__(self):
            self.action_space = Env2()


    class Env2(object):

        def __init__(self):
            self.flat_dim = 2


    env_spec = Env1()
    test = OUStrategy(env_spec)

    states = []
    for i in range(1000):
        states.append(test.evolve_state()[0])
    import matplotlib.pyplot as plt

    plt.plot(states)
    plt.show()


