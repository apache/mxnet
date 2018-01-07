import time
import math
import numpy as np
import torch

from log_uniform import LogUniformSampler

def log_uniform_sample(N, size):
    log_N = math.log(N)
    x = torch.Tensor(size).uniform_(0, 1)
    value = torch.exp(x * log_N).long() - 1
    return torch.remainder(value, N)

def log_uniform(class_id, range_max):
    return (math.log(class_id+2) - math.log(class_id+1)) / math.log(range_max+1)

def log_uniform_distribution(range_max):
    distribution = np.asarray([log_uniform(idx, range_max) for idx in range(range_max)])
    return torch.from_numpy(distribution)

N = 793471
num_samples = 8192
batch_size = 128 * 20

start_time = time.time()
log_uniform_sample(N, num_samples)
end_time = time.time()
print("non_unique log_uniform numpy", end_time - start_time)

p = log_uniform_distribution(N)
start_time = time.time()
torch.multinomial(p, num_samples, replacement=True)
end_time = time.time()
print("non_unique multinomial cuda", end_time - start_time)

start_time = time.time()
torch.multinomial(p, 100)
end_time = time.time()
print("unique multinomial cuda", end_time - start_time)

sampler = LogUniformSampler(N)
labels = np.random.choice(N, batch_size)

start_time = time.time()
sample_id, true_freq, sample_freq = sampler.sample(num_samples, labels)
end_time = time.time()
print("unique log_uniform c++", end_time - start_time)

"""
sampler = LogUniformSampler()
start_time = time.time()
sample_id = sampler.sample(N, num_samples, unique=True, labels=labels.tolist())
end_time = time.time()
print("unique no_accidental_hits log_uniform c++", end_time - start_time)

label_set = set(labels.tolist())
for idx in sample_id:
    assert(idx not in label_set)
"""
