from libcpp.unordered_set cimport unordered_set
from libcpp.vector cimport vector
from libcpp.pair cimport pair
cimport numpy as np
import cython

cdef extern from "Log_Uniform_Sampler.h":
    cdef cppclass Log_Uniform_Sampler:
        Log_Uniform_Sampler(int) except +
        vector[float] expected_count(int, vector[long]) except +
        vector[pair[long, long]] accidental_matches(vector[long], vector[long]) except +
        unordered_set[long] sample(int, int*) except +
        unordered_set[long] sample_unique(int, unordered_set[long]) except +
        float probability(int) except +

cdef class LogUniformSampler:
    cdef Log_Uniform_Sampler* c_sampler

    def __cinit__(self, N):
        self.c_sampler = new Log_Uniform_Sampler(N)

    def __dealloc__(self):
        del self.c_sampler

    def sample_unique(self, size, labels):
        samples = list(self.c_sampler.sample_unique(size, labels))
        return samples

    def sample(self, size, labels):
        cdef int num_tries
        samples = list(self.c_sampler.sample(size, &num_tries))
        true_freq = self.c_sampler.expected_count(num_tries, labels.tolist())
        sample_freq = self.c_sampler.expected_count(num_tries, samples)
        return samples, true_freq, sample_freq

    def accidental_match(self, labels, samples):
        return self.c_sampler.accidental_matches(labels, samples)

    def probability(self, idx):
        return self.c_sampler.probability(idx)
