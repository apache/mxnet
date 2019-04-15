"""
A collection of different ways to calibrate noise to privacy
"""
import numpy as np
from math import exp, sqrt
from scipy.special import erf
from scipy.optimize import brentq

from autodp import rdp_acct, rdp_bank, dp_acct, dp_bank


# Subsampling lemma and its inverse

def subsample_epsdelta(eps,delta,prob):
    """
    :param eps: privacy loss eps of the base mechanism
    :param delta: privacy loss delta of the base mechanism
    :param prob: subsampling probability
    :return: Amplified eps and delta

    This result applies to both subsampling with replacement and Poisson subsampling.

    The result for Poisson subsmapling is due to Theorem 1 of :
    Li, Ninghui, Qardaji, Wahbeh, and Su, Dong. On sampling, anonymization, and differential privacy or,
    k-anonymization meets differential privacy

    The result for Subsampling with replacement is due to:
    Jon Ullman's lecture notes: http://www.ccs.neu.edu/home/jullman/PrivacyS17/HW1sol.pdf
    See the proof of (b)

    """

    return np.log(1+prob*(np.exp(eps)-1)), prob*delta


def subsample_epsdelta_inverse(eps,delta,prob):
    # Give a target subsampled epsilon and delta, and subsampling probability. Calibrate the base eps, delta
    assert(prob > 0 and prob <=1)
    return np.log((np.exp(eps)-1)/prob + 1), np.minimum(delta/prob,1.0)


def subsample_epsdelta_get_prob(eps_target,delta_target, eps_base,delta_base):
    """
    Calibrate the probability of subsampling
    :param eps_target: Target eps in subsampled mechanisms
    :param delta_target: Target delta in subsampled mechanisms
    :param eps_base: base eps, in subsampled mechanisms
    :param delta_base: base delta in subsampled mechanisms.
    :return: subsampling probability  prob
    """
    return np.mininum(1.0,np.minimum(delta_target/delta_base,  (np.exp(eps_target)-1)/(np.exp(eps_base)-1)))



# we start with a general calibration function.

def RDP_mech(rdp_func, eps, delta, param_name, params, bounds=[0,np.inf],k=1,prob=1.0):
    # Take an analytical RDP, find the smallest noise level to achieve (eps, delta)-DP.
    """
    :param rdp_func: the RDP function that takes in params and alpha like those in 'rdp_bank'.
    :param eps:  the required eps
    :param delta:  the required delta
    :param param_name
    :param params: a template dictionary to modify from.
    :param bounds: a pair of numbers indicating the valid ranges of the parameters
    :return: params_out:  the calibrated params.
    """
    assert (eps > 0 and delta > 0)


    def func(x):
        # We assume that the rdp_func and param_name is chosen such that this function is either monotonically
        # increasing or decreasing.
        params[param_name] = x
        rdp = lambda alpha: rdp_func(params, alpha)
        tmp_acct = rdp_acct.anaRDPacct()

        if prob < 1.0 and prob >0:
            tmp_acct.compose_subsampled_mechanism(rdp, prob,coeff=k)
        else:
            tmp_acct.compose_mechanism(rdp,coeff=k)

        eps_tmp = tmp_acct.get_eps(delta)
        return eps_tmp - eps

    # Project to feasible region
    a=np.minimum(bounds[1], np.maximum(1.0, bounds[0]))
    b=np.maximum(bounds[0], np.minimum(2.0, bounds[1]))
    maxiter = 100
    count = 1
    # find a valid range
    while np.sign(func(a)) == np.sign(func(b)):
        a = np.maximum(a/2,bounds[0])
        b = np.minimum(b*2,bounds[1])
        count = count + 1
        if count >=maxiter:
            # infeasible
            return None

    root = brentq(func, a, b)

    # assign calibarated results
    params_out = params
    params_out[param_name] = root

    return params_out

def subsampled_RDP_mech_get_prob(rdp_func, eps, delta, params,k=1):
    """
    This function calibrates the probability of subsampling to achieve a prescribed privacy goal.
    :param rdp_func: rdp of the base mechanism (as in those in rdp_bank)
    :param eps:  the required eps
    :param delta:  the required delta
    :param params: the parameter object for the first argument, rdp_func
    :param k: (optional) number of rounds to compose.
    :return:
    """
    assert (eps > 0 and delta > 0)
    def func(x):
        rdp = lambda alpha: rdp_func(params, alpha)
        tmp_acct = rdp_acct.anaRDPacct()
        tmp_acct.compose_subsampled_mechanism(rdp, x, coeff=k)
        eps_tmp = tmp_acct.get_eps(delta)
        return eps_tmp - eps

    root = brentq(func, 0, 1)
    return root


def gaussian_mech(eps,delta,k=1,prob=1.0):
    """
    Calibrate the scale parameter b of the Gaussian mechanism using RDP

    :param eps: prescribed eps
    :param delta: prescribed delta
    :param k: (optional) number of times to run this mechanism.
    :return: the parameter structure for this randomized algorithm
    """
    assert(eps>0)

    params = {}
    param_name = 'sigma'
    params = RDP_mech(rdp_bank.RDP_gaussian, eps, delta, param_name, params,k=k,prob=prob)

    return params



def laplace_mech(eps, delta, k=1, prob=1.0):
    """
    Calibrate the scale parameter b of the Laplace mechanism

    :param eps: prescribed eps
    :param delta: prescribed delta
    :param k: (optional) number of times to run this mechanism.
    :return: the parameter structure for this randomized algorithm
    """
    assert(eps>0 and delta >= 0)

    binf = 1.0/eps*k
    if delta == 0:
        params = {'b':binf}
    else:
        params = {}
        param_name = 'b'
        params = RDP_mech(rdp_bank.RDP_laplace, eps, delta, param_name, params, k=k,prob=prob)
        if params['b'] > binf:
            # further take maximum w.r.t. alpha = infty
            params['b'] = binf

    return params


def randresponse_mech(eps, delta, k=1,prob=1.0):
    """
    Calibrate the bernoulli parameter p of the randomized response mechanisms

    :param eps: prescribed eps
    :param delta: prescribed delta
    :param k: (optional) number of times to run this mechanism.
    :return: the parameter structure for this randomized algorithm
    """
    assert (eps > 0 and delta >= 0)

    pinf= np.exp(1.0*eps/k)/(1+np.exp(1.0*eps/k))
    if delta ==1:
        return {'p':1}
    if delta == 0:
        params = {'p':pinf}
    else:
        params = {}
        param_name = 'p'
        params = RDP_mech(rdp_bank.RDP_randresponse, eps, delta, param_name, params,
                          bounds=[np.exp(1.0*eps/k/2)/(1+np.exp(1.0*eps/2/k)),1-1e-8],k=k,prob=prob)
        if params['p'] < pinf:
            # further take maximum w.r.t. alpha = infty
            params['p'] = pinf

    return params


def classical_gaussian_mech(eps,delta):
    """
    The classical gaussian mechanism. For benchmarking purposes only.
    DO NOT USE in practice as it is dominated by `ana_gaussian_mech' and `gaussian_mech`.

    :param eps: prescribed 0< eps <1
    :param delta: prescribed 0 < delta <1
    :return: required noise level.
    """
    assert(eps > 0 and eps <=1),\
        "The classical Gaussian mechanism only supports 0 < eps <1, try `gaussian_mech` and `ana_gaussian_mech`"
    if delta <= 0:
        return np.inf
    if delta >= 1:
        return 0
    params = {'sigma': 1.0 * np.sqrt(2 * np.log(1.25 / delta)) / eps}
    return params


def ana_gaussian_mech(epsilon, delta, tol=1.e-12):
    """ Calibrate a Gaussian perturbation for differential privacy using the analytic Gaussian mechanism of [Balle and Wang, ICML'18]

    Modified from https://github.com/BorjaBalle/analytic-gaussian-mechanism/blob/master/agm-example.py

    Arguments:
    epsilon : target epsilon (epsilon > 0)
    delta : target delta (0 < delta < 1)
    tol : error tolerance for binary search (tol > 0)
    Output:
    params : a dictionary that contains field `sigma' --- the standard deviation of Gaussian noise needed to achieve
        (epsilon,delta)-DP under global sensitivity 1
    """

    def Phi(t):
        return 0.5*(1.0 + erf(float(t)/sqrt(2.0)))

    def caseA(epsilon,s):
        return Phi(sqrt(epsilon*s)) - exp(epsilon)*Phi(-sqrt(epsilon*(s+2.0)))

    def caseB(epsilon,s):
        return Phi(-sqrt(epsilon*s)) - exp(epsilon)*Phi(-sqrt(epsilon*(s+2.0)))

    def doubling_trick(predicate_stop, s_inf, s_sup):
        while(not predicate_stop(s_sup)):
            s_inf = s_sup
            s_sup = 2.0*s_inf
        return s_inf, s_sup

    def binary_search(predicate_stop, predicate_left, s_inf, s_sup):
        s_mid = s_inf + (s_sup-s_inf)/2.0
        while(not predicate_stop(s_mid)):
            if (predicate_left(s_mid)):
                s_sup = s_mid
            else:
                s_inf = s_mid
            s_mid = s_inf + (s_sup-s_inf)/2.0
        return s_mid

    delta_thr = caseA(epsilon, 0.0)

    if (delta == delta_thr):
        alpha = 1.0
        sigma = alpha / sqrt(2.0 * epsilon)

    else:
        if (delta > delta_thr):
            predicate_stop_DT = lambda s : caseA(epsilon, s) >= delta
            function_s_to_delta = lambda s : caseA(epsilon, s)
            predicate_left_BS = lambda s : function_s_to_delta(s) > delta
            function_s_to_alpha = lambda s : sqrt(1.0 + s/2.0) - sqrt(s/2.0)

        else:
            predicate_stop_DT = lambda s : caseB(epsilon, s) <= delta
            function_s_to_delta = lambda s : caseB(epsilon, s)
            predicate_left_BS = lambda s : function_s_to_delta(s) < delta
            function_s_to_alpha = lambda s : sqrt(1.0 + s/2.0) + sqrt(s/2.0)

        predicate_stop_BS = lambda s : abs(function_s_to_delta(s) - delta) <= tol

        s_inf, s_sup = doubling_trick(predicate_stop_DT, 0.0, 1.0)
        s_final = binary_search(predicate_stop_BS, predicate_left_BS, s_inf, s_sup)
        alpha = function_s_to_alpha(s_final)
        sigma = alpha/sqrt(2.0*epsilon)


    params = {'sigma': sigma}
    return params


