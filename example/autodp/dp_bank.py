"""
This module contains a collections of the inverse of `privacy_calibrator'.  Given a fixed randomized algorithm and a
desired parameter `delta` it calculates the corresponding (eps,delta)-DP guarantee.

These are building blocks of many differential privacy algorithms.

In some cases, given a fixed randomized algorithm on a fixed data set, it calculates the corresponding (eps,delta)-pDP.
"""

import numpy as np
from autodp import rdp_acct, rdp_bank


def get_eps_rdp(func, delta):
    """
    This is the generic function that uses RDP accountant and RDP function to solve for eps given delta
    :param func:
    :param delta:
    :return: The corresponding epsilon
    """
    assert(delta >= 0)
    acct = rdp_acct.anaRDPacct(m=10,m_max=10)
    acct.compose_mechanism(func)
    return acct.get_eps(delta)


def get_eps_rdp_subsampled(func, delta, prob):
    """
    This is the generic function that uses RDP accountant and RDP function to solve for eps given delta
    :param func:
    :param delta:
    :return: The corresponding epsilon
    """
    assert(delta >= 0)
    assert(prob >=0)
    if prob==0:
        return 0
    elif prob == 1:
        return get_eps_rdp(func,delta)
    else:
        acct = rdp_acct.anaRDPacct()
        acct.compose_subsampled_mechanism(func,prob)
        return acct.get_eps(delta)


# Get the eps and delta for a single Gaussian mechanism
def get_eps_gaussian(sigma, delta):
    """ This function calculates the eps for Gaussian Mech given sigma and delta"""
    assert(delta >= 0)
    func = lambda x: rdp_bank.RDP_gaussian({'sigma':sigma},x)
    return get_eps_rdp(func,delta)


def get_eps_ana_gaussian(sigma, delta):
    """ TBA"""
    #TODO: add the analytical guassian way to calculating epsilon given delta.
    return None


def get_eps_laplace(b,delta):
    assert(delta >= 0)
    func = lambda x: rdp_bank.RDP_laplace({'b':b},x)
    return get_eps_rdp(func,delta)


def get_eps_randresp(p,delta):
    assert(delta >= 0)
    func = lambda x: rdp_bank.RDP_randresponse({'p':p},x)
    return get_eps_rdp(func, delta)

