"""Wrapper for training network with different algorithms."""
from mxnet import gluon
from mxnet import autograd
from block.sampler import NaiveSampler
from block.matcher import CompositeMatcher, BipartiteMatcher, MaximumMatcher
from block.coder import MultiClassEncoder, MultiClassDecoder
from block.coder import NormalizedBoxCenterEncoder, NormalizedBoxCenterDecoder



def train_net(net, trainer)
