#!/usr/bin/env python
# -*- coding: utf-8 -*-
import functools
import contextlib
import random

random.seed()

class Array():
    def __init__(self, name):
        self._name = name

    def rename(self, name):
        self._name = name

    def __add__(self, other):
        if jit_enabled:
            jit_sequence.append(('add', self, other))
            print('add delayed')
        else:
            print('add eager')
        if grad_enabled:
            grad_sequence.append(('add', self, other))
        return Array('({} + {})'.format(self, other))

    def __mul__(self, other):
        if jit_enabled:
            jit_sequence.append(('mul', self, other))
            print('mul delayed')
        else:
            print('mul eager')
        if grad_enabled:
            grad_sequence.append(('mul', self, other))
        return Array('({} * {})'.format(self, other))

    def __repr__(self):
        return 'Array {}'.format(self._name)

    def eval(self):
        if jit_enabled:
            # guard instruction
            flush_jit_sequence()

    def __getitem__(self, key):
        self.eval()
        return random.randint(0, 1)


jit_enabled = False
jit_sequence = []
grad_enabled = False
grad_sequence = []

jit_cache = {}
def flush_jit_sequence():
    k = tuple(map(lambda i: (i[0], i[1]._name, i[2]._name), jit_sequence))
    if k in jit_cache:
        execute(jit_cache[k])
    else:
        # Run asynchronously
        seq = optimize(jit_sequence)
        jit_cache[k] = seq
        # Run in main thread
        execute(jit_sequence)
    jit_sequence.clear()

def flush_grad_sequence():
    g = get_grad(grad_sequence)
    jit_sequence.extend(g)
    grad_sequence.clear()

def reset_jit_cache():
    jit_cache.clear()

# Part of NNVM.
def execute(seq):
    print('executing seq {}'.format(seq))

def optimize(seq):
    return 'optimized {}'.format(seq)

def get_grad(seq):
    return list(map(lambda i: (i[0] + '_grad', i[1], i[2]), reversed(seq)))


@contextlib.contextmanager
def jit():
    global jit_enabled
    jit_enabled = True
    yield
    flush_jit_sequence()
    jit_enabled = False

@contextlib.contextmanager
def grad():
    global grad_enabled
    grad_enabled = True
    yield
    flush_grad_sequence()
    grad_enabled = False