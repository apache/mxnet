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

# pylint: skip-file

from mxnet import np
from mxnet.test_utils import same, assert_almost_equal, rand_shape_nd, use_np
import numpy as _np
import scipy as sp

import pytest

from sklearn.utils._testing import assert_allclose

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.decomposition._pca import _assess_dimension
from sklearn.decomposition._pca import _infer_dimension

iris = datasets.load_iris()
PCA_SOLVERS = ['full', 'arpack', 'randomized', 'auto']


@use_np
@pytest.mark.parametrize('svd_solver', PCA_SOLVERS)
@pytest.mark.parametrize('n_components', range(1, iris.data.shape[1]))
def test_pca(svd_solver, n_components):
    X = np.array(iris.data)
    pca = PCA(n_components=n_components, svd_solver=svd_solver)

    # check the shape of fit.transform
    X_r = pca.fit(X).transform(X)
    print(type(X_r))
    assert X_r.shape[1] == n_components

    # check the equivalence of fit.transform and fit_transform
    X_r2 = pca.fit_transform(X)
    assert_allclose(X_r, X_r2, atol=1e-5, rtol=1e-5)
    X_r = pca.transform(X)
    assert_allclose(X_r, X_r2, atol=1e-5, rtol=1e-5)

    # Test get_covariance and get_precision
    cov = np.array(pca.get_covariance())
    precision = np.array(pca.get_precision())
    assert_allclose(np.dot(cov, precision), np.eye(X.shape[1]), rtol=1e-4, atol=1e-4)


@use_np
def test_no_empty_slice_warning():
    # test if we avoid numpy warnings for computing over empty arrays
    n_components = 10
    n_features = n_components + 2  # anything > n_comps triggered it in 0.16
    X = np.random.uniform(-1, 1, size=(n_components, n_features))
    pca = PCA(n_components=n_components)
    with pytest.warns(None) as record:
        pca.fit(X)
    assert not record.list


@pytest.mark.parametrize('copy', [True, False])
@pytest.mark.parametrize('solver', PCA_SOLVERS)
@use_np
def test_whitening(solver, copy):
    # Check that PCA output has unit-variance
    n_samples = 100
    n_features = 80
    n_components = 30
    rank = 50

    # some low rank data with correlated features
    X = np.dot(np.random.randn(n_samples, rank),
               np.dot(np.diag(np.linspace(10.0, 1.0, rank)),
                      np.random.randn(rank, n_features)))
    # the component-wise variance of the first 50 features is 3 times the
    # mean component-wise variance of the remaining 30 features
    X[:, :50] *= 3

    assert X.shape == (n_samples, n_features)

    # the component-wise variance is thus highly varying:
    assert X.std(axis=0).std() > 35

    # whiten the data while projecting to the lower dim subspace
    X_ = X.copy()  # make sure we keep an original across iterations.
    pca = PCA(n_components=n_components, whiten=True, copy=copy,
              svd_solver=solver, random_state=0, iterated_power=7)
    # test fit_transform
    X_whitened = pca.fit_transform(X_.copy())
    assert X_whitened.shape == (n_samples, n_components)
    X_whitened2 = pca.transform(X_)
    assert_allclose(X_whitened, X_whitened2, rtol=1e-3, atol=1e-3)

    assert_allclose(X_whitened.std(ddof=1, axis=0), np.ones(n_components), atol=1e-5, rtol=1e-5)
    assert_allclose(
        X_whitened.mean(axis=0), np.zeros(n_components), atol=1e-5, rtol=1e-5
    )

    X_ = X.copy()
    pca = PCA(n_components=n_components, whiten=False, copy=copy,
              svd_solver=solver).fit(X_)
    X_unwhitened = pca.transform(X_)
    assert X_unwhitened.shape == (n_samples, n_components)


@pytest.mark.parametrize('svd_solver', ['arpack', 'randomized'])
@use_np
def test_pca_explained_variance_equivalence_solver(svd_solver):
    n_samples, n_features = 100, 80
    X = np.random.randn(n_samples, n_features)

    pca_full = PCA(n_components=2, svd_solver='full')
    pca_other = PCA(n_components=2, svd_solver=svd_solver, random_state=0)

    pca_full.fit(X)
    pca_other.fit(X)

    assert_allclose(
        pca_full.explained_variance_,
        pca_other.explained_variance_,
        rtol=5e-2
    )
    assert_allclose(
        pca_full.explained_variance_ratio_,
        pca_other.explained_variance_ratio_,
        rtol=5e-2
    )


@pytest.mark.parametrize("svd_solver", ['arpack', 'randomized'])
@use_np
def test_pca_singular_values_consistency(svd_solver):
    n_samples, n_features = 100, 80
    X = np.random.randn(n_samples, n_features)

    pca_full = PCA(n_components=2, svd_solver='full')
    pca_other = PCA(n_components=2, svd_solver=svd_solver)

    pca_full.fit(X)
    pca_other.fit(X)

    assert_allclose(
        pca_full.singular_values_, pca_other.singular_values_, rtol=5e-3
    )


@pytest.mark.parametrize("svd_solver", PCA_SOLVERS)
@use_np
def test_pca_singular_values(svd_solver):
    n_samples, n_features = 100, 80
    X = np.random.randn(n_samples, n_features)

    pca = PCA(n_components=2, svd_solver=svd_solver)
    X_trans = np.array(pca.fit_transform(X))

    # compare to the Frobenius norm
    assert_allclose(
        np.sum(np.array(pca.singular_values_ ** 2)), np.linalg.norm(X_trans, "fro") ** 2,
        atol=1e-5, rtol=1e-5
    )
    # Compare to the 2-norms of the score vectors
    assert_allclose(
        pca.singular_values_, np.sqrt(np.sum(X_trans ** 2, axis=0)),
        atol=1e-5, rtol=1e-5
    )

    # set the singular values and see what er get back
    n_samples, n_features = 100, 110
    X = np.random.randn(n_samples, n_features)

    pca = PCA(n_components=3, svd_solver=svd_solver)
    X_trans = np.array(pca.fit_transform(X))
    X_trans /= np.sqrt(np.sum(X_trans ** 2, axis=0))
    X_trans[:, 0] *= 3.142
    X_trans[:, 1] *= 2.718
    X_hat = np.dot(X_trans, np.array(pca.components_))
    pca.fit(X_hat)
    assert_allclose(pca.singular_values_, [3.142, 2.718, 1.0], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("svd_solver", PCA_SOLVERS)
@use_np
def test_pca_check_projection(svd_solver):
    # Test that the projection of data is correct
    n, p = 100, 3
    X = np.random.randn(n, p) * .1
    X[:10] += np.array([3, 4, 5])
    Xt = 0.1 * np.random.randn(1, p) + np.array([3, 4, 5])

    Yt = PCA(n_components=2, svd_solver=svd_solver).fit(X).transform(Xt)
    Yt /= np.sqrt((Yt ** 2).sum())

    assert_allclose(np.abs(Yt[0][0]), 1., rtol=5e-3)


@pytest.mark.parametrize("svd_solver", PCA_SOLVERS)
@use_np
def test_pca_check_projection_list(svd_solver):
    # Test that the projection of data is correct
    X = [[1.0, 0.0], [0.0, 1.0]]
    pca = PCA(n_components=1, svd_solver=svd_solver, random_state=0)
    X_trans = pca.fit_transform(X)
    assert X_trans.shape, (2, 1)
    assert_allclose(X_trans.mean(), 0.00, atol=1e-12)
    assert_allclose(X_trans.std(), 0.71, rtol=5e-3)
