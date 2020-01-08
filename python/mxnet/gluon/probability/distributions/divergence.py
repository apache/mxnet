# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# coding: utf-8
# pylint: disable=wildcard-import
"""KL divergence functions."""
__all__ = ['register_kl', 'kl_divergence']
from .utils import getF


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
        if not hasattr(_KL_storage, func_name):
            setattr(_KL_storage, func_name, func)
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
    func_impl = getattr(_KL_storage, func_name, None)
    if (not callable(func_impl)):
        raise NotImplementedError(
            "KL divergence between {} and {} is not implemented.".format(type_p, type_q))
    return func_impl


class _KL_storage():
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
