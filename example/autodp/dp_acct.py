"""
This file implements the classical DP accountant by Kairouz, Oh, V.
"""


import numpy as np
from scipy.optimize import minimize_scalar

from autodp import utils

import math


class DP_acct:
    """
    A class that keeps track of (eps,delta) of all mechanisms that got run so far.
    Then it allows delta => epsilon, and epsilon => delta queries.
    """
    #
    #DPlosses = []

    def __init__(self,disp=False):
        self.DPlosses = []
        self.eps_state1 = 0
        self.eps_state2 = 0
        self.eps_state3 = 0
        self.delta_state = 0
        self.delta_state2 = 0
        self.disp=disp

    def update_DPlosses(self,eps,delta):
        self.DPlosses.append([eps,delta])
        self.eps_state1 += eps
        self.eps_state2 += (np.exp(eps) - 1) * eps / (np.exp(eps) + 1)
        self.eps_state3 += eps ** 2
        self.delta_state += np.log(1-delta)
        self.delta_state2 += delta
        #update the optimal DPlosses here?

    def get_eps_delta_naive(self):
        return self.eps_state1, self.delta_state2

    def get_minimum_possible_delta(self):
        return 1-np.exp(self.delta_state)


    def get_eps(self,delta):
        """ make use of KOV'15 to calculate the composition for hetereogeneous mechanisms"""
        # TODO: further improve upon this with Mutagh & Vadhan's approximation algorithm
        assert(delta >= 0)
        if delta>=1:
            return 0
        if delta is 0: # asking for pure DP
            if self.delta_state2 is 0:
                return self.delta_state2
            else:
                return np.inf
                #if self.disp:
                #    print("Pure DP is not feabile. Choose non-zero delta")
                #return -1

        # 1-delta =  (1-deltatilde) exp(state)
        deltatilde = 1 - np.exp(np.log(1-delta) -self.delta_state)
        if deltatilde <= 0:
            return np.inf
            #if self.disp:
            #    print("The chosen delta is not feasible. delta needs to be at least ", 1-np.exp(self.delta_state))
            #return -1

        eps1 = self.eps_state1
        eps2 = self.eps_state2 + (self.eps_state3 * 2 * np.log(np.exp(1) + self.eps_state3 ** 0.5 / deltatilde))**0.5
        eps3 = self.eps_state2 + (2*self.eps_state3*np.log(1 / deltatilde))**0.5

        return np.minimum(np.minimum(eps1,eps2),eps3)