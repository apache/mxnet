"""
A collection of analytical expressions of the Renyi Differential Privacy for popular randomized algorithms.
All expressions in this file takes in a function of the RDP order alpha and output the corresponding RDP.

These are used to create symbolic randomized functions for the RDP accountant to track.

Some of the functions contain the renyi divergence of two given distributions, these are useful to keep track of
the per-instance RDP associated with two given data sets.
"""


import numpy as np

from autodp import utils



def RDP_gaussian(params, alpha):
    """
    :param params:
        'sigma' --- is the normalized noise level: std divided by global L2 sensitivity
    :param alpha: The order of the Renyi Divergence
    :return: Evaluation of the RDP's epsilon
    """
    sigma = params['sigma']
    # assert(sigma > 0)
    # assert(alpha >= 0)
    return 0.5 / sigma ** 2 * alpha


def RDP_laplace(params, alpha):
    """
    :param params:
        'b' --- is the is the ratio of the scale parameter and L1 sensitivity
    :param alpha: The order of the Renyi Divergence
    :return: Evaluation of the RDP's epsilon
    """

    b = params['b']
    # assert(b > 0)
    # assert(alpha >= 0)
    alpha=1.0*alpha
    if alpha <= 1:
        return (1 / b + np.exp(-1 / b) - 1)
    elif np.isinf(alpha):
        return 1/b
    else:  # alpha > 1
        return utils.stable_logsumexp_two((alpha-1.0) / b + np.log(alpha / (2.0 * alpha - 1)),
                                           -1.0*alpha / b + np.log((alpha-1.0) / (2.0 * alpha - 1)))/(alpha-1)


def RDP_randresponse(params, alpha):
    """
    :param params:
        'p' --- is the Bernoulli probability p of outputting the truth
    :param alpha: The order of the Renyi Divergence
    :return: Evaluation of the RDP's epsilon
    """

    p = params['p']
    assert((p >= 0) and (p <= 1))
    # assert(alpha >= 0)
    if p == 1 or p == 0:
        return np.inf
    if alpha <= 1:
        return (2 * p - 1) * np.log(p / (1 - p))
    elif np.isinf(alpha):
        return np.abs(np.log((1.0*p/(1-p))))
    else:  # alpha > 1
        return utils.stable_logsumexp_two(alpha * np.log(p) + (1 - alpha) * np.log(1 - p),
                                           alpha * np.log(1 - p) + (1 - alpha) * np.log(p))/(alpha-1)



def RDP_expfamily(params, alpha):
    """
    :param params: 'Delta': max distance of the natural parameters between two adjacent data sets in a certain norms.
        'L' 'B' are lambda functions. They are upper bounds of the local smoothness and local Lipschitzness
        of the log-partition function A, as a function of the radius of the local neighborhood in that norm.
    :param alpha: The order of the Renyi Divergence
    :return: Evaluation of the RDP's epsilon

        See Proposition 29 and Remark 30 of Wang, Balle, Kasiviswanathan (2018)
    """
    Delta = params['Delta']
    L  = params['L'] # Local smoothness function that takes radius kappa as a input.
    B = params['B']  # Local Lipschitz function that takes radius kappa as a input.

    return np.minimum(alpha * L(alpha*Delta) * Delta ** 2,
                      (B((alpha-1)*Delta) + B(Delta)) * Delta)


def pRDP_diag_gaussian(params, alpha):
    """
    :param params:
        'mu1', 'mu2', 'sigma1', 'sigma2', they are all d-dimensional numpy arrays
    :param alpha: The order of the Renyi Divergence
    :return: Evaluation of the pRDP's epsilon

    See page 27 of http://mast.queensu.ca/~communications/Papers/gil-msc11.pdf for reference.
    """
    # the parameter is mu1, mu2, sigma1, sigma2
    # they are all d-dimensional numpy arrays
    # Everything can be generalized to general covariance, but
    # calculating A and checking the positive semidefiniteness of A is tricky

    mu1 = params['mu1']
    mu2 = params['mu2']
    sigma1 = params['sigma1']
    sigma2 = params['sigma2']

    def extrapolate(a, b):
        return alpha * a +(1 - alpha) * b

    #A = extrapolate(sigma1 ** (-1), sigma2 ** (-1))

    sigma = extrapolate(sigma2, sigma1)

    if not (sigma > 0).all():
        return np.inf
    else:
        #sigma = extrapolate(sigma2, sigma1)
        return (alpha / 2.0 * np.dot((mu1 - mu2),  (mu1 - mu2) / sigma) - 0.5 / (alpha-1) *
                (np.sum(np.log(sigma)) - extrapolate(np.sum(np.log(sigma2)), np.sum(np.log(sigma1)))))


def RDP_pureDP(params,alpha):
    """
    This function generically converts pure DP to Renyi DP.
    It implements Lemma 1.4 of Bun et al.'s CDP paper.
    With an additional cap at eps.

    :param params: pure DP parameter
    :param alpha: The order of the Renyi Divergence
    :return:Evaluation of the RDP's epsilon
    """
    eps = params['eps']
    assert(eps>=0)
    if alpha < 1:
        # Pure DP needs to have identical support, thus - log(q(p>0)) = 0.
        return 0
    else:
        return np.minimum(eps,alpha*eps*eps/2)


def RDP_subsampled_pureDP(params, alpha):
    """
    The special function for approximating the privacy amplification by subsampling for pure DP mechanism.
    1. This is more or less tight
    2. This applies for both Poisson subsampling (with add-remove DP definition)
       and Subsampling (with replace DP Definition)
    3. It evaluates in O(1)

    :param params: pure DP parameter, (optional) second order RDP, and alpha.
    :param alpha: The order of the Renyi Divergence
    :return: Evaluation of the RDP's epsilon

    """
    eps = params['eps']
    #eps2 = params['eps2'] # this parameter is optional, if unknown just use min(eps,2*eps*eps/2)
    if 'eps2' in params.keys():
        eps2= params['eps2']
    else:
        eps2 = np.minimum(eps,eps*eps)
    prob = params['prob']
    assert((prob<1) and (prob >= 0))
    assert(eps >= 0 and eps2 >=0)


    def rdp_int(x):
        if x == np.inf:
            return eps
        s, mag = utils.stable_log_diff_exp(eps,0)
        s, mag2 = utils.stable_log_diff_exp(eps2,0)

        s, mag3 = utils.stable_log_diff_exp(x*utils.stable_logsumexp_two(np.log(1-prob),np.log(prob)+eps),
                                            np.log(x) + np.log(prob) + mag)
        s, mag4 = utils.stable_log_diff_exp(mag3, np.log(1.0*x/2)+np.log(x-1)+2*np.log(prob)
                                            + np.log( np.exp(2*mag) - np.exp(np.min([mag,2*mag,mag2]))))

        return 1/(x-1)*mag4


        ## The following is the implementation of the second line.
        # if x <= 2:
        #     # Just return rdp 2
        #     return utils.stable_logsumexp([0, np.log(1.0*2/2)+np.log(2-1)+2*np.log(prob)
        #                                    + np.min([mag2,mag,2*mag])])
        # else:
        #     return 1/(x-1)*utils.stable_logsumexp([0, np.log(1.0*x/2)+np.log(x-1)+2*np.log(prob)
        #                                            + np.min([mag2,mag,2*mag]),
        #                                           3*np.log(prob) + 3*mag + np.log(x) + np.log(x-1)
        #                                           + np.log(x-2) - np.log(6) +
        #                                           (x-3) * utils.stable_logsumexp_two(np.log(1-prob),np.log(prob)+eps)])

    if alpha < 1:
        return 0
    else:
        return utils.RDP_linear_interpolation(rdp_int, alpha)


def pRDP_asymp_subsampled_gaussian(params, alpha):
    """
    :param params:
    :param alpha: The order of the Renyi Divergence
    :return: Evaluation of the pRDP's epsilon

    See Example 19 of Wang, Balle, Kasiviswanathan (2018)
    """
    sigma = params['sigma']
    prob = params['prob']
    assert((prob<1) and (prob >= 0))

    # The example where we have an approximately worst case asymptotic data set
    thresh = sigma**2/prob + 1
    if alpha <= 1:
        return 0
    elif alpha >= thresh:
        return np.inf
    else:
        return (prob ** 2 / (2*sigma**2) * alpha * (thresh-1) / (thresh - alpha)
                + np.log((thresh-1)/thresh) / 2
                ) + np.log((thresh-1) / (thresh - alpha)) / 2 /(alpha-1)


def pRDP_asymp_subsampled_gaussian_best_case(params, alpha):
    """
    :param params:
    :param alpha: The order of the Renyi Divergence
    :return: Evaluation of the pRDP's epsilon

    See Example 20 of Wang, Balle, Kasiviswanathan (2018)
    """
    sigma = params['sigma']
    prob = params['prob']
    n = params['n']
    assert((prob<1) and (prob >= 0))

    # The example where we have an approximately best case data set
    return prob**2 / (2*sigma**2 + prob*(n-1.0/n)/2) * alpha



def pRDP_expfamily(params, alpha):
    """
    :param params:
        'eta1', 'eta2' are the natural parameters of the exp family distributions.
        'A' is a lambda function handle the log partition. `A' needs to handle cases of infinity.
        'mu' is the mean of the sufficient statistics from distribution 1.
    :param alpha: The order of the Renyi Divergence
    :return: Evaluation of the pRDP's epsilon
    """
    # Theorem 1.2.19 of http://mast.queensu.ca/~communications/Papers/gil-msc11.pdf
    eta1 = params['eta1']
    eta2 = params['eta2']
    A = params['A']
    mu = params['mu'] # This is used only for calculating KL divergence
    # mu is also the gradient of A at eta1.


    assert(alpha >= 1)

    if alpha == 1:
        return np.dot(eta1-eta2, mu) + A(eta1) - A(eta2)

    def extrapolate(a, b):
        return alpha * a + (1 - alpha) * b

    tmp = A(extrapolate(eta1, eta2))
    if np.isinf(tmp) or np.isnan(tmp):
        return np.inf
    else: # alpha > 1
        return (A(extrapolate(eta1, eta2)) - extrapolate(A(eta1), A(eta2))) / (alpha - 1)