import numpy as np
import numpy.random as npr

import mxnet.gluon as gluon

K = 5
N = 1000

# Get a random probability vector.
probs = npr.dirichlet(np.ones(K), 1).ravel()

# Construct the table.
alias_method_sampler = gluon.data.AliasMethodSampler(K, probs)

# Generate variates.
X = alias_method_sampler.draw(N)

# check sampled probabilities
sampled_probs = [float(x)/N for x in X]

print(probs)
print(sampled_probs)


