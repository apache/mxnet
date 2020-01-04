from mxnet import np, npx
from .utils import getF

__all__ = ['register_kl', 'kl_divergence']


def register_kl(typeP, typeQ):
    """Decorator for registering custom implementation
    of kl divergence between distribution `typeP` and `typeQ`

    Returns
    -------
    wrapped function
    """
    func_name = "_kl_" + str(typeP.__name__) \
                + "_" + str(typeQ.__name__)

    def decorator(func):
        func_arg_num = func.__code__.co_argcount
        if (func_arg_num != 2):
            raise TypeError('Expect kl_divergence implementation '
                            + 'to have exactly two arguments, but got {}'.format(func_arg_num))
        if not hasattr(kl_storage, func_name):
            setattr(kl_storage, func_name, func)
        else:
            # Behavior TBD.
            print("Error: Duplicate definition")
        return func
    return decorator


def kl_divergence(p, q):
    r"""Return the kl divergence between p and q,
        this method will automatically dispatch
        to the corresponding function based on q's type.

        Parameters
        ----------
        p : Distribution
            lhs distribution.
        q : Distribution
            rhs distribution.

        Returns
        -------
        Tensor
            KL(self||q)
        """
    func = _dispatch_kl(p.__class__.__name__, q.__class__.__name__)
    return func(p, q)


def _dispatch_kl(type_p, type_q):
    r"""KL divergence methods should be registered
    with distribution name,
    i.e. the implementation of KL(P(\theta)||Q(\theta))
    should be named after _kl_{P}_{Q}

    Parameters
    ----------
    type_q : Typename of a distribution
    type_q : Typename of a distribution


    Returns
    -------
    Get a class method with function name.
    """
    func_name = "_kl_" + str(type_p) + "_" + str(type_q)
    return getattr(kl_storage, func_name)


class kl_storage():
    r"""Class for storing the definition of kl divergence 
    between distributions.
    All the class methods should be static
    """

    @staticmethod
    def _kl_Normal_Normal(p, q):
        F = getF(p, q)
        var_ratio = (p.scale / q.scale) ** 2
        t1 = ((p.loc - q.loc) / q.scale) ** 2
        return 0.5 * (var_ratio + t1 - 1 - F.np.log(var_ratio))
