import numpy as np


class ReplayMem(object):

    def __init__(
        self, 
        obs_dim,
        act_dim,
        memory_size=1000000):

        # allocate space for memory cells
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.obss = np.zeros((memory_size, obs_dim))
        self.acts = np.zeros((memory_size, act_dim))
        self.rwds = np.zeros((memory_size, ))
        self.ends = np.zeros(memory_size, dtype='uint8')
        self.memory_size = memory_size
        self.top = -1
        self.size = 0

    def add_sample(self, obs, act, rwd, end):

        self.top = (self.top + 1) % self.memory_size
        self.obss[self.top] = obs
        self.acts[self.top] = act
        self.rwds[self.top] = rwd
        self.ends[self.top] = end

        if self.size < self.memory_size:
            self.size += 1

    def get_batch(self, batch_size):

        assert self.size >= batch_size

        indices = np.zeros(batch_size, dtype="uint64")
        transit_indices = np.zeros(batch_size, dtype="uint64")

        counter = 0
        while counter < batch_size:
            idx = np.random.randint(0, self.size)
            # case where the last piece of memory is sampled
            # which does not have a successor state
            if idx == self.top:
                    continue
            transit_idx = (idx + 1) % self.memory_size
            indices[counter] = idx
            transit_indices[counter] = transit_idx
            counter += 1

        return (self.obss[indices],
                self.acts[indices],
                self.rwds[indices],
                self.ends[indices],
                self.obss[transit_indices])


if __name__ == "__main__":

    memory = ReplayMem(2, 1, memory_size=10)
    memory.add_sample(np.array([2, 2]), np.array([2]), 10, 0)
    memory.add_sample(np.array([2, 2]), np.array([2]), 10, 0)
    memory.add_sample(np.array([2, 2]), np.array([2]), 10, 0)
    memory.add_sample(np.array([2, 2]), np.array([2]), 10, 0)
    memory.add_sample(np.array([2, 2]), np.array([2]), 10, 0)
    memory.add_sample(np.array([2, 2]), np.array([2]), 10, 0)
    memory.add_sample(np.array([2, 2]), np.array([2]), 10, 0)
    memory.add_sample(np.array([2, 2]), np.array([2]), 10, 0)
    memory.add_sample(np.array([1, 1]), np.array([1]), 100, 1)
    print memory.obss
    print memory.acts 
    print memory.rwds 
    print memory.ends   
    print memory.get_batch(5)

