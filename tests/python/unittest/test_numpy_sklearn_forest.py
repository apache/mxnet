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

import mxnet as mx
from mxnet import np
import numpy as _np

import pytest
import pickle
import math
from collections import defaultdict
import itertools
from itertools import combinations
from itertools import product
from typing import Dict, Any

from scipy.special import comb

from sklearn.utils._testing import assert_array_almost_equal
from sklearn.utils._testing import assert_array_equal
from sklearn.utils._testing import assert_raises
from sklearn.utils._testing import assert_warns
from sklearn.utils._testing import assert_warns_message
from sklearn.utils._testing import ignore_warnings
from sklearn.utils._testing import skip_if_no_parallel

from sklearn.exceptions import NotFittedError

from sklearn import datasets
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.utils.validation import check_random_state

from mxnet.test_utils import same, assert_almost_equal, rand_shape_nd, use_np


# toy sample
X = np.array([[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]])
y = np.array([-1, -1, -1, 1, 1, 1])
T = np.array([[-1, -1], [2, 2], [3, 2]])
true_result = np.array([-1, 1, 1])

# Larger classification sample used for testing feature importances
X_large, y_large = datasets.make_classification(
    n_samples=500, n_features=10, n_informative=3, n_redundant=0,
    n_repeated=0, shuffle=False, random_state=0)

# also load the iris dataset
# and randomly permute it
iris = datasets.load_iris()
rng = check_random_state(0)
perm = rng.permutation(iris.target.size)
iris.data = np.array(iris.data[perm])
iris.target = np.array(iris.target[perm])

# Make regression dataset
X_reg, y_reg = datasets.make_regression(n_samples=500, n_features=10,
                                        random_state=1)
X_reg, y_reg = np.array(X_reg), np.array(y_reg)

# also make a hastie_10_2 dataset
hastie_X, hastie_y = datasets.make_hastie_10_2(n_samples=20, random_state=1)
hastie_X = np.array(hastie_X).astype(np.float32)
hastie_y = np.array(hastie_y)

FOREST_CLASSIFIERS = {
    "ExtraTreesClassifier": ExtraTreesClassifier,
    "RandomForestClassifier": RandomForestClassifier,
}

FOREST_REGRESSORS = {
    "ExtraTreesRegressor": ExtraTreesRegressor,
    "RandomForestRegressor": RandomForestRegressor,
}

FOREST_TRANSFORMERS = {
    "RandomTreesEmbedding": RandomTreesEmbedding,
}

FOREST_ESTIMATORS: Dict[str, Any] = dict()
FOREST_ESTIMATORS.update(FOREST_CLASSIFIERS)
FOREST_ESTIMATORS.update(FOREST_REGRESSORS)
FOREST_ESTIMATORS.update(FOREST_TRANSFORMERS)

FOREST_CLASSIFIERS_REGRESSORS: Dict[str, Any] = FOREST_CLASSIFIERS.copy()
FOREST_CLASSIFIERS_REGRESSORS.update(FOREST_REGRESSORS)


@use_np
@pytest.mark.parametrize('name', FOREST_CLASSIFIERS)
def test_classification_toy(name):
    """Check classification on a toy dataset."""
    ForestClassifier = FOREST_CLASSIFIERS[name]

    clf = ForestClassifier(n_estimators=10, random_state=1)
    clf.fit(X, y)
    assert_array_equal(clf.predict(T), true_result)
    assert 10 == len(clf)

    clf = ForestClassifier(n_estimators=10, max_features=1, random_state=1)
    clf.fit(X, y)
    assert_array_equal(clf.predict(T), true_result)
    assert 10 == len(clf)

    # also test apply
    leaf_indices = clf.apply(X)
    assert leaf_indices.shape == (len(X), clf.n_estimators)


@use_np
@pytest.mark.parametrize('name', FOREST_CLASSIFIERS)
@pytest.mark.parametrize('criterion', ("gini", "entropy"))
def test_iris_criterion(name, criterion):
    # Check consistency on dataset iris.
    ForestClassifier = FOREST_CLASSIFIERS[name]

    clf = ForestClassifier(n_estimators=10, criterion=criterion,
                           random_state=1)
    clf.fit(iris.data, iris.target)
    score = clf.score(iris.data, iris.target)
    assert score > 0.9, ("Failed with criterion %s and score = %f"
                         % (criterion, score))

    clf = ForestClassifier(n_estimators=10, criterion=criterion,
                           max_features=2, random_state=1)
    clf.fit(iris.data, iris.target)
    score = clf.score(iris.data, iris.target)
    assert score > 0.5, ("Failed with criterion %s and score = %f"
                         % (criterion, score))


@use_np
@pytest.mark.parametrize('name', FOREST_REGRESSORS)
@pytest.mark.parametrize('criterion', ("mse", "mae", "friedman_mse"))
def test_regression_criterion(name, criterion):
    # Check consistency on regression dataset.
    ForestRegressor = FOREST_REGRESSORS[name]

    reg = ForestRegressor(n_estimators=5, criterion=criterion,
                          random_state=1)
    reg.fit(X_reg, y_reg)
    score = reg.score(X_reg, y_reg)
    assert score > 0.93, ("Failed with max_features=None, criterion %s "
                          "and score = %f" % (criterion, score))

    reg = ForestRegressor(n_estimators=5, criterion=criterion,
                          max_features=6, random_state=1)
    reg.fit(X_reg, y_reg)
    score = reg.score(X_reg, y_reg)
    assert score > 0.92, ("Failed with max_features=6, criterion %s "
                          "and score = %f" % (criterion, score))


@use_np
@pytest.mark.parametrize('name', FOREST_REGRESSORS)
def check_regressor_attributes(name):
    # Regression models should not have a classes_ attribute.
    r = FOREST_REGRESSORS[name](random_state=0)
    assert not hasattr(r, "classes_")
    assert not hasattr(r, "n_classes_")

    r.fit([[1, 2, 3], [4, 5, 6]], [1, 2])
    assert not hasattr(r, "classes_")
    assert not hasattr(r, "n_classes_")


@use_np
@pytest.mark.parametrize('name', FOREST_CLASSIFIERS)
def test_probability(name):
    # Predict probabilities.
    ForestClassifier = FOREST_CLASSIFIERS[name]
    with _np.errstate(divide="ignore"):
        clf = ForestClassifier(n_estimators=10, random_state=1, max_features=1,
                               max_depth=1)
        clf.fit(iris.data, iris.target)
        assert_array_almost_equal(_np.sum(clf.predict_proba(iris.data), axis=1),
                                  np.ones(iris.data.shape[0]))
        assert_array_almost_equal(clf.predict_proba(iris.data),
                                  _np.exp(clf.predict_log_proba(iris.data)))


@use_np
@pytest.mark.parametrize('dtype', (np.float64, np.float32))
@pytest.mark.parametrize(
        'name, criterion',
        itertools.chain(product(FOREST_CLASSIFIERS,
                                ["gini", "entropy"]),
                        product(FOREST_REGRESSORS,
                                ["mse", "friedman_mse", "mae"])))
def test_importances(name, criterion, dtype):
    tolerance = 0.01
    if name in FOREST_REGRESSORS and criterion == "mae":
        tolerance = 0.05
    # cast as dype
    X = X_large.astype(dtype, copy=False)
    y = y_large.astype(dtype, copy=False)

    ForestEstimator = FOREST_ESTIMATORS[name]

    est = ForestEstimator(n_estimators=10, criterion=criterion,
                          random_state=0)
    est.fit(X, y)
    importances = est.feature_importances_

    # The forest estimator can detect that only the first 3 features of the
    # dataset are informative:
    n_important = _np.sum(importances > 0.1)
    assert importances.shape[0] == 10
    assert n_important == 3
    assert _np.all(importances[:3] > 0.1)

    # Check with parallel
    importances = est.feature_importances_
    est.set_params(n_jobs=2)
    importances_parallel = est.feature_importances_
    assert_array_almost_equal(importances, importances_parallel)

    # Check with sample weights
    sample_weight = check_random_state(0).randint(1, 10, len(X))
    est = ForestEstimator(n_estimators=10, random_state=0, criterion=criterion)
    est.fit(X, y, sample_weight=sample_weight)
    importances = est.feature_importances_
    assert _np.all(importances >= 0.0)

    for scale in [0.5, 100]:
        est = ForestEstimator(n_estimators=10, random_state=0,
                              criterion=criterion)
        est.fit(X, y, sample_weight=scale * sample_weight)
        importances_bis = est.feature_importances_
        assert _np.abs(importances - importances_bis).mean() < tolerance


def check_oob_score(name, X, y, n_estimators=20):
    # Check that oob prediction is a good estimation of the generalization
    # error.

    # Proper behavior
    est = FOREST_ESTIMATORS[name](oob_score=True, random_state=0,
                                  n_estimators=n_estimators, bootstrap=True)
    n_samples = X.shape[0]
    est.fit(X[:n_samples // 2, :], y[:n_samples // 2])
    test_score = est.score(X[n_samples // 2:, :], y[n_samples // 2:])
    oob_score = est.oob_score_

    assert abs(test_score - oob_score) < 0.1 and oob_score > 0.7

    # Check warning if not enough estimators
    with _np.errstate(divide="ignore", invalid="ignore"):
        est = FOREST_ESTIMATORS[name](oob_score=True, random_state=0,
                                      n_estimators=1, bootstrap=True)
        assert_warns(UserWarning, est.fit, X, y)


@use_np
@pytest.mark.parametrize('name', FOREST_CLASSIFIERS)
def test_oob_score_classifiers(name):
    check_oob_score(name, iris.data, iris.target)

    # non-contiguous targets in classification
    check_oob_score(name, iris.data, iris.target * 2 + 1)


@use_np
@pytest.mark.parametrize('name', FOREST_REGRESSORS)
def test_oob_score_regressors(name):
    check_oob_score(name, X_reg, y_reg, 50)


def check_oob_score_raise_error(name):
    ForestEstimator = FOREST_ESTIMATORS[name]

    if name in FOREST_TRANSFORMERS:
        for oob_score in [True, False]:
            assert_raises(TypeError, ForestEstimator, oob_score=oob_score)

        assert_raises(NotImplementedError, ForestEstimator()._set_oob_score,
                      X, y)

    else:
        # Unfitted /  no bootstrap / no oob_score
        for oob_score, bootstrap in [(True, False), (False, True),
                                     (False, False)]:
            est = ForestEstimator(oob_score=oob_score, bootstrap=bootstrap,
                                  random_state=0)
            assert not hasattr(est, "oob_score_")

        # No bootstrap
        assert_raises(ValueError, ForestEstimator(oob_score=True,
                                                  bootstrap=False).fit, X, y)


@use_np
@pytest.mark.parametrize('name', FOREST_ESTIMATORS)
def test_oob_score_raise_error(name):
    check_oob_score_raise_error(name)


def check_gridsearch(name):
    forest = FOREST_CLASSIFIERS[name]()
    clf = GridSearchCV(forest, {'n_estimators': (1, 2), 'max_depth': (1, 2)})
    clf.fit(iris.data, iris.target)


@use_np
@pytest.mark.parametrize('name', FOREST_CLASSIFIERS)
def test_gridsearch(name):
    # Check that base trees can be grid-searched.
    check_gridsearch(name)


def check_parallel(name, X, y):
    """Check parallel computations in classification"""
    ForestEstimator = FOREST_ESTIMATORS[name]
    forest = ForestEstimator(n_estimators=10, n_jobs=3, random_state=0)

    forest.fit(X, y)
    assert len(forest) == 10

    forest.set_params(n_jobs=1)
    y1 = forest.predict(X)
    forest.set_params(n_jobs=2)
    y2 = forest.predict(X)
    assert_array_almost_equal(y1, y2, 3)


@use_np
@pytest.mark.parametrize('name', FOREST_CLASSIFIERS_REGRESSORS)
def test_parallel(name):
    if name in FOREST_CLASSIFIERS:
        X = iris.data
        y = iris.target
    elif name in FOREST_REGRESSORS:
        X = X_reg
        y = y_reg

    check_parallel(name, X, y)


def check_pickle(name, X, y):
    # Check pickability.

    ForestEstimator = FOREST_ESTIMATORS[name]
    obj = ForestEstimator(random_state=0)
    obj.fit(X, y)
    score = obj.score(X, y)
    pickle_object = pickle.dumps(obj)

    obj2 = pickle.loads(pickle_object)
    assert type(obj2) == obj.__class__
    score2 = obj2.score(X, y)
    assert score == score2


@use_np
@pytest.mark.parametrize('name', FOREST_CLASSIFIERS_REGRESSORS)
def test_pickle(name):
    if name in FOREST_CLASSIFIERS:
        X = iris.data
        y = iris.target
    elif name in FOREST_REGRESSORS:
        X = X_reg
        y = y_reg

    check_pickle(name, X[::2], y[::2])


def check_multioutput(name):
    # Check estimators on multi-output problems.

    X_train = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1], [-2, 1],
               [-1, 1], [-1, 2], [2, -1], [1, -1], [1, -2]]
    y_train = [[-1, 0], [-1, 0], [-1, 0], [1, 1], [1, 1], [1, 1], [-1, 2],
               [-1, 2], [-1, 2], [1, 3], [1, 3], [1, 3]]
    X_test = [[-1, -1], [1, 1], [-1, 1], [1, -1]]
    y_test = [[-1, 0], [1, 1], [-1, 2], [1, 3]]

    est = FOREST_ESTIMATORS[name](random_state=0, bootstrap=False)
    y_pred = est.fit(X_train, y_train).predict(X_test)
    assert_array_almost_equal(y_pred, y_test)

    if name in FOREST_CLASSIFIERS:
        with _np.errstate(divide="ignore"):
            proba = est.predict_proba(X_test)
            assert len(proba) == 2
            assert proba[0].shape == (4, 2)
            assert proba[1].shape == (4, 4)

            log_proba = est.predict_log_proba(X_test)
            assert len(log_proba) == 2
            assert log_proba[0].shape == (4, 2)
            assert log_proba[1].shape == (4, 4)


@use_np
@pytest.mark.parametrize('name', FOREST_CLASSIFIERS_REGRESSORS)
def test_multioutput(name):
    check_multioutput(name)


@use_np
@pytest.mark.parametrize('name', FOREST_CLASSIFIERS)
def test_multioutput_string(name):
    # Check estimators on multi-output problems with string outputs.

    X_train = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1], [-2, 1],
               [-1, 1], [-1, 2], [2, -1], [1, -1], [1, -2]]
    y_train = [["red", "blue"], ["red", "blue"], ["red", "blue"],
               ["green", "green"], ["green", "green"], ["green", "green"],
               ["red", "purple"], ["red", "purple"], ["red", "purple"],
               ["green", "yellow"], ["green", "yellow"], ["green", "yellow"]]
    X_test = [[-1, -1], [1, 1], [-1, 1], [1, -1]]
    y_test = [["red", "blue"], ["green", "green"],
              ["red", "purple"], ["green", "yellow"]]

    est = FOREST_ESTIMATORS[name](random_state=0, bootstrap=False)
    y_pred = est.fit(X_train, y_train).predict(X_test)
    assert_array_equal(y_pred, y_test)

    with _np.errstate(divide="ignore"):
        proba = est.predict_proba(X_test)
        assert len(proba) == 2
        assert proba[0].shape == (4, 2)
        assert proba[1].shape == (4, 4)

        log_proba = est.predict_log_proba(X_test)
        assert len(log_proba) == 2
        assert log_proba[0].shape == (4, 2)
        assert log_proba[1].shape == (4, 4)


def check_classes_shape(name):
    # Test that n_classes_ and classes_ have proper shape.
    ForestClassifier = FOREST_CLASSIFIERS[name]

    # Classification, single output
    clf = ForestClassifier(random_state=0).fit(X, y)

    assert clf.n_classes_ == 2
    assert_array_equal(clf.classes_, [-1, 1])

    # Classification, multi-output
    print(type(y), type(np.array(y) * 2))
    _y = _np.vstack((y, np.array(y) * 2)).T
    clf = ForestClassifier(random_state=0).fit(X, _y)

    assert_array_equal(clf.n_classes_, [2, 2])
    assert_array_equal(clf.classes_, [[-1, 1], [-2, 2]])


@use_np
@pytest.mark.parametrize('name', FOREST_CLASSIFIERS)
def test_classes_shape(name):
    check_classes_shape(name)


@use_np
def test_random_trees_dense_type():
    # Test that the `sparse_output` parameter of RandomTreesEmbedding
    # works by returning a dense array.

    # Create the RTE with sparse=False
    hasher = RandomTreesEmbedding(n_estimators=10, sparse_output=False)
    X, y = datasets.make_circles(factor=0.5)
    X, y = np.array(X), np.array(y)
    X_transformed = hasher.fit_transform(X)

    # Assert that type is ndarray, not scipy.sparse.csr.csr_matrix
    assert type(X_transformed) == _np.ndarray


@use_np
def test_random_trees_dense_equal():
    # Test that the `sparse_output` parameter of RandomTreesEmbedding
    # works by returning the same array for both argument values.

    # Create the RTEs
    hasher_dense = RandomTreesEmbedding(n_estimators=10, sparse_output=False,
                                        random_state=0)
    hasher_sparse = RandomTreesEmbedding(n_estimators=10, sparse_output=True,
                                         random_state=0)
    X, y = datasets.make_circles(factor=0.5)
    X, y = np.array(X), np.array(y)
    X_transformed_dense = hasher_dense.fit_transform(X)
    X_transformed_sparse = hasher_sparse.fit_transform(X)

    # Assert that dense and sparse hashers have same array.
    assert_array_equal(X_transformed_sparse.toarray(), X_transformed_dense)


@use_np
def test_random_hasher():
    # test random forest hashing on circles dataset
    # make sure that it is linearly separable.
    # even after projected to two SVD dimensions
    # Note: Not all random_states produce perfect results.
    hasher = RandomTreesEmbedding(n_estimators=30, random_state=1)
    X, y = datasets.make_circles(factor=0.5)
    X, y = np.array(X), np.array(y)
    X_transformed = hasher.fit_transform(X)

    # test fit and transform:
    hasher = RandomTreesEmbedding(n_estimators=30, random_state=1)
    assert_array_equal(hasher.fit(X).transform(X).toarray(),
                       X_transformed.toarray())

    # one leaf active per data point per forest
    assert X_transformed.shape[0] == X.shape[0]
    assert_array_equal(X_transformed.sum(axis=1), hasher.n_estimators)
    svd = TruncatedSVD(n_components=2)
    X_reduced = svd.fit_transform(X_transformed)
    linear_clf = LinearSVC()
    linear_clf.fit(X_reduced, y)
    assert linear_clf.score(X_reduced, y) == 1.


@use_np
def test_parallel_train():
    rng = check_random_state(12321)
    n_samples, n_features = 80, 30
    X_train = rng.randn(n_samples, n_features)
    y_train = rng.randint(0, 2, n_samples)

    clfs = [
        RandomForestClassifier(n_estimators=20, n_jobs=n_jobs,
                               random_state=12345).fit(X_train, y_train)
        for n_jobs in [1, 2, 3, 8, 16, 32]
    ]

    X_test = rng.randn(n_samples, n_features)
    probas = [clf.predict_proba(X_test) for clf in clfs]
    for proba1, proba2 in zip(probas, probas[1:]):
        assert_array_almost_equal(proba1, proba2)


@use_np
def test_distribution():
    rng = check_random_state(12321)

    # Single variable with 4 values
    X = rng.randint(0, 4, size=(1000, 1))
    y = rng.rand(1000)
    n_trees = 500

    reg = ExtraTreesRegressor(n_estimators=n_trees, random_state=42).fit(X, y)

    uniques = defaultdict(int)
    for tree in reg.estimators_:
        tree = "".join(("%d,%d/" % (f, int(t)) if f >= 0 else "-")
                       for f, t in zip(tree.tree_.feature,
                                       tree.tree_.threshold))

        uniques[tree] += 1

    uniques = sorted([(1. * count / n_trees, tree)
                      for tree, count in uniques.items()])

    # On a single variable problem where X_0 has 4 equiprobable values, there
    # are 5 ways to build a random tree. The more compact (0,1/0,0/--0,2/--) of
    # them has probability 1/3 while the 4 others have probability 1/6.

    assert len(uniques) == 5
    assert 0.20 > uniques[0][0]  # Rough approximation of 1/6.
    assert 0.20 > uniques[1][0]
    assert 0.20 > uniques[2][0]
    assert 0.20 > uniques[3][0]
    assert uniques[4][0] > 0.3
    assert uniques[4][1] == "0,1/0,0/--0,2/--"

    # Two variables, one with 2 values, one with 3 values
    X = np.empty((1000, 2))
    X[:, 0] = np.random.randint(0, 2, 1000)
    X[:, 1] = np.random.randint(0, 3, 1000)
    y = rng.rand(1000)

    reg = ExtraTreesRegressor(max_features=1, random_state=1).fit(X, y)

    uniques = defaultdict(int)
    for tree in reg.estimators_:
        tree = "".join(("%d,%d/" % (f, int(t)) if f >= 0 else "-")
                       for f, t in zip(tree.tree_.feature,
                                       tree.tree_.threshold))

        uniques[tree] += 1

    uniques = [(count, tree) for tree, count in uniques.items()]
    assert len(uniques) == 8


@use_np
@pytest.mark.parametrize('name', FOREST_ESTIMATORS)
def test_max_leaf_nodes_max_depth(name):
    X, y = hastie_X, hastie_y

    # Test precedence of max_leaf_nodes over max_depth.
    ForestEstimator = FOREST_ESTIMATORS[name]
    est = ForestEstimator(max_depth=1, max_leaf_nodes=4,
                          n_estimators=1, random_state=0).fit(X, y)
    assert est.estimators_[0].get_depth() == 1

    est = ForestEstimator(max_depth=1, n_estimators=1,
                          random_state=0).fit(X, y)
    assert est.estimators_[0].get_depth() == 1


@use_np
@pytest.mark.parametrize('name', FOREST_ESTIMATORS)
def test_min_samples_split(name):
    X, y = hastie_X, hastie_y
    ForestEstimator = FOREST_ESTIMATORS[name]

    # test boundary value
    assert_raises(ValueError,
                  ForestEstimator(min_samples_split=-1).fit, X, y)
    assert_raises(ValueError,
                  ForestEstimator(min_samples_split=0).fit, X, y)
    assert_raises(ValueError,
                  ForestEstimator(min_samples_split=1.1).fit, X, y)

    est = ForestEstimator(min_samples_split=10, n_estimators=1, random_state=0)
    est.fit(X, y)
    node_idx = est.estimators_[0].tree_.children_left != -1
    node_samples = est.estimators_[0].tree_.n_node_samples[node_idx]

    assert _np.min(node_samples) > len(X) * 0.5 - 1, (
        "Failed with {0}".format(name))

    est = ForestEstimator(min_samples_split=0.5, n_estimators=1,
                          random_state=0)
    est.fit(X, y)
    node_idx = est.estimators_[0].tree_.children_left != -1
    node_samples = est.estimators_[0].tree_.n_node_samples[node_idx]

    assert _np.min(node_samples) > len(X) * 0.5 - 1, (
        "Failed with {0}".format(name))


@use_np
@pytest.mark.parametrize('name', FOREST_CLASSIFIERS_REGRESSORS)
@pytest.mark.parametrize('dtype', (np.float64, np.float32))
def test_memory_layout(name, dtype):
    # Check that it works no matter the memory layout

    est = FOREST_ESTIMATORS[name](random_state=0, bootstrap=False)

    # Nothing
    X = np.array(_np.asarray(iris.data, dtype=dtype))
    y = np.array(iris.target)
    assert_array_almost_equal(est.fit(X, y).predict(X), y)

    # C-order
    X = np.array(_np.asarray(iris.data, order="C", dtype=dtype))
    y = iris.target
    assert_array_almost_equal(est.fit(X, y).predict(X), y)


@use_np
@pytest.mark.parametrize('name', FOREST_ESTIMATORS)
def test_1d_input(name):
    X = iris.data[:, 0]
    X_2d = iris.data[:, 0].reshape((-1, 1))
    y = iris.target

    with ignore_warnings():
        ForestEstimator = FOREST_ESTIMATORS[name]
        assert_raises(ValueError, ForestEstimator(n_estimators=1,
                                                  random_state=0).fit, X, y)

        est = ForestEstimator(random_state=0)
        est.fit(X_2d, y)

        if name in FOREST_CLASSIFIERS or name in FOREST_REGRESSORS:
            assert_raises(ValueError, est.predict, X)


@use_np
@pytest.mark.parametrize('name', FOREST_CLASSIFIERS)
def check_class_weights(name):
    # Check class_weights resemble sample_weights behavior.
    ForestClassifier = FOREST_CLASSIFIERS[name]

    # Iris is balanced, so no effect expected for using 'balanced' weights
    clf1 = ForestClassifier(random_state=0)
    clf1.fit(iris.data, iris.target)
    clf2 = ForestClassifier(class_weight='balanced', random_state=0)
    clf2.fit(iris.data, iris.target)
    assert_almost_equal(clf1.feature_importances_, clf2.feature_importances_)

    # Make a multi-output problem with three copies of Iris
    iris_multi = _np.vstack((iris.target, iris.target, iris.target)).T
    # Create user-defined weights that should balance over the outputs
    clf3 = ForestClassifier(class_weight=[{0: 2., 1: 2., 2: 1.},
                                          {0: 2., 1: 1., 2: 2.},
                                          {0: 1., 1: 2., 2: 2.}],
                            random_state=0)
    clf3.fit(iris.data, iris_multi)
    assert_almost_equal(clf2.feature_importances_, clf3.feature_importances_)
    # Check against multi-output "balanced" which should also have no effect
    clf4 = ForestClassifier(class_weight='balanced', random_state=0)
    clf4.fit(iris.data, iris_multi)
    assert_almost_equal(clf3.feature_importances_, clf4.feature_importances_)

    # Inflate importance of class 1, check against user-defined weights
    sample_weight = np.ones(iris.target.shape)
    sample_weight[iris.target == 1] *= 100
    class_weight = {0: 1., 1: 100., 2: 1.}
    clf1 = ForestClassifier(random_state=0)
    clf1.fit(iris.data, iris.target, sample_weight)
    clf2 = ForestClassifier(class_weight=class_weight, random_state=0)
    clf2.fit(iris.data, iris.target)
    assert_almost_equal(clf1.feature_importances_, clf2.feature_importances_)

    # Check that sample_weight and class_weight are multiplicative
    clf1 = ForestClassifier(random_state=0)
    clf1.fit(iris.data, iris.target, sample_weight ** 2)
    clf2 = ForestClassifier(class_weight=class_weight, random_state=0)
    clf2.fit(iris.data, iris.target, sample_weight)
    assert_almost_equal(clf1.feature_importances_, clf2.feature_importances_)


@use_np
@pytest.mark.parametrize('name', FOREST_CLASSIFIERS)
def test_class_weight_balanced_and_bootstrap_multi_output(name):
    # Test class_weight works for multi-output"""
    ForestClassifier = FOREST_CLASSIFIERS[name]
    _y = _np.vstack((y, np.array(y) * 2)).T
    clf = ForestClassifier(class_weight='balanced', random_state=0)
    clf.fit(X, _y)
    clf = ForestClassifier(class_weight=[{-1: 0.5, 1: 1.}, {-2: 1., 2: 1.}],
                           random_state=0)
    clf.fit(X, _y)
    # smoke test for balanced subsample
    clf = ForestClassifier(class_weight='balanced_subsample', random_state=0)
    clf.fit(X, _y)


@use_np
@pytest.mark.parametrize('name', FOREST_CLASSIFIERS)
def test_class_weight_errors(name):
    # Test if class_weight raises errors and warnings when expected.
    ForestClassifier = FOREST_CLASSIFIERS[name]
    _y = _np.vstack((y, np.array(y) * 2)).T

    # Invalid preset string
    clf = ForestClassifier(class_weight='the larch', random_state=0)
    assert_raises(ValueError, clf.fit, X, y)
    assert_raises(ValueError, clf.fit, X, _y)

    # Warning warm_start with preset
    clf = ForestClassifier(class_weight='balanced', warm_start=True,
                           random_state=0)
    assert_warns(UserWarning, clf.fit, X, y)
    assert_warns(UserWarning, clf.fit, X, _y)

    # Not a list or preset for multi-output
    clf = ForestClassifier(class_weight=1, random_state=0)
    assert_raises(ValueError, clf.fit, X, _y)

    # Incorrect length list for multi-output
    clf = ForestClassifier(class_weight=[{-1: 0.5, 1: 1.}], random_state=0)
    assert_raises(ValueError, clf.fit, X, _y)


@use_np
@pytest.mark.parametrize('name', FOREST_ESTIMATORS)
def test_warm_start(name, random_state=42):
    # Test if fitting incrementally with warm start gives a forest of the
    # right size and the same results as a normal fit.
    X, y = hastie_X, hastie_y
    ForestEstimator = FOREST_ESTIMATORS[name]
    est_ws = None
    for n_estimators in [5, 10]:
        if est_ws is None:
            est_ws = ForestEstimator(n_estimators=n_estimators,
                                     random_state=random_state,
                                     warm_start=True)
        else:
            est_ws.set_params(n_estimators=n_estimators)
        est_ws.fit(X, y)
        assert len(est_ws) == n_estimators

    est_no_ws = ForestEstimator(n_estimators=10, random_state=random_state,
                                warm_start=False)
    est_no_ws.fit(X, y)

    assert (set([tree.random_state for tree in est_ws]) ==
            set([tree.random_state for tree in est_no_ws]))

    assert_array_equal(est_ws.apply(X), est_no_ws.apply(X),
                       err_msg="Failed with {0}".format(name))


@use_np
@pytest.mark.parametrize('name', FOREST_ESTIMATORS)
def test_warm_start_clear(name):
    # Test if fit clears state and grows a new forest when warm_start==False.
    X, y = hastie_X, hastie_y
    ForestEstimator = FOREST_ESTIMATORS[name]
    est = ForestEstimator(n_estimators=5, max_depth=1, warm_start=False,
                          random_state=1)
    est.fit(X, y)

    est_2 = ForestEstimator(n_estimators=5, max_depth=1, warm_start=True,
                            random_state=2)
    est_2.fit(X, y)  # inits state
    est_2.set_params(warm_start=False, random_state=1)
    est_2.fit(X, y)  # clears old state and equals est

    assert_array_almost_equal(est_2.apply(X), est.apply(X))


@use_np
@pytest.mark.parametrize('name', FOREST_ESTIMATORS)
def test_warm_start_smaller_n_estimators(name):
    # Test if warm start second fit with smaller n_estimators raises error.
    X, y = hastie_X, hastie_y
    ForestEstimator = FOREST_ESTIMATORS[name]
    est = ForestEstimator(n_estimators=5, max_depth=1, warm_start=True)
    est.fit(X, y)
    est.set_params(n_estimators=4)
    assert_raises(ValueError, est.fit, X, y)


@use_np
@pytest.mark.parametrize('name', FOREST_ESTIMATORS)
def test_warm_start_equal_n_estimators(name):
    # Test if warm start with equal n_estimators does nothing and returns the
    # same forest and raises a warning.
    X, y = hastie_X, hastie_y
    ForestEstimator = FOREST_ESTIMATORS[name]
    est = ForestEstimator(n_estimators=5, max_depth=3, warm_start=True,
                          random_state=1)
    est.fit(X, y)

    est_2 = ForestEstimator(n_estimators=5, max_depth=3, warm_start=True,
                            random_state=1)
    est_2.fit(X, y)
    # Now est_2 equals est.

    est_2.set_params(random_state=2)
    assert_warns(UserWarning, est_2.fit, X, y)
    # If we had fit the trees again we would have got a different forest as we
    # changed the random state.
    assert_array_equal(est.apply(X), est_2.apply(X))


@use_np
@pytest.mark.parametrize('name', FOREST_CLASSIFIERS_REGRESSORS)
def test_warm_start_oob(name):
    # Test that the warm start computes oob score when asked.
    X, y = hastie_X, hastie_y
    ForestEstimator = FOREST_ESTIMATORS[name]
    # Use 15 estimators to avoid 'some inputs do not have OOB scores' warning.
    est = ForestEstimator(n_estimators=15, max_depth=3, warm_start=False,
                          random_state=1, bootstrap=True, oob_score=True)
    est.fit(X, y)

    est_2 = ForestEstimator(n_estimators=5, max_depth=3, warm_start=False,
                            random_state=1, bootstrap=True, oob_score=False)
    est_2.fit(X, y)

    est_2.set_params(warm_start=True, oob_score=True, n_estimators=15)
    est_2.fit(X, y)

    assert hasattr(est_2, 'oob_score_')
    assert est.oob_score_ == est_2.oob_score_

    # Test that oob_score is computed even if we don't need to train
    # additional trees.
    est_3 = ForestEstimator(n_estimators=15, max_depth=3, warm_start=True,
                            random_state=1, bootstrap=True, oob_score=False)
    est_3.fit(X, y)
    assert not hasattr(est_3, 'oob_score_')

    est_3.set_params(oob_score=True)
    ignore_warnings(est_3.fit)(X, y)

    assert est.oob_score_ == est_3.oob_score_


@use_np
def test_dtype_convert(n_classes=15):
    classifier = RandomForestClassifier(random_state=0, bootstrap=False)

    X = np.eye(n_classes)
    y = [ch for ch in 'ABCDEFGHIJKLMNOPQRSTU'[:n_classes]]

    result = classifier.fit(X, y).predict(X)
    assert_array_equal(classifier.classes_, y)
    assert_array_equal(result, y)


@use_np
@pytest.mark.parametrize('name', FOREST_CLASSIFIERS_REGRESSORS)
def test_decision_path(name):
    X, y = hastie_X, hastie_y
    n_samples = X.shape[0]
    ForestEstimator = FOREST_ESTIMATORS[name]
    est = ForestEstimator(n_estimators=5, max_depth=1, warm_start=False,
                          random_state=1)
    est.fit(X, y)
    indicator, n_nodes_ptr = est.decision_path(X)

    assert indicator.shape[1] == n_nodes_ptr[-1]
    assert indicator.shape[0] == n_samples
    assert_array_equal(_np.diff(n_nodes_ptr),
                       [e.tree_.node_count for e in est.estimators_])

    # Assert that leaves index are correct
    leaves = est.apply(X)
    for est_id in range(leaves.shape[1]):
        leave_indicator = [indicator[i, n_nodes_ptr[est_id] + j]
                           for i, j in enumerate(leaves[:, est_id])]
        assert_array_almost_equal(leave_indicator, np.ones(shape=n_samples))


@use_np
def test_min_impurity_split():
    # Test if min_impurity_split of base estimators is set
    # Regression test for #8006
    X, y = datasets.make_hastie_10_2(n_samples=100, random_state=1)
    X, y = np.array(X), np.array(y)
    all_estimators = [RandomForestClassifier, RandomForestRegressor,
                      ExtraTreesClassifier, ExtraTreesRegressor]

    for Estimator in all_estimators:
        est = Estimator(min_impurity_split=0.1)
        est = assert_warns_message(FutureWarning,
                                   "min_impurity_decrease",
                                   est.fit, X, y)
        for tree in est.estimators_:
            assert tree.min_impurity_split == 0.1


@use_np
def test_min_impurity_decrease():
    X, y = datasets.make_hastie_10_2(n_samples=100, random_state=1)
    X, y = np.array(X), np.array(y)
    all_estimators = [RandomForestClassifier, RandomForestRegressor,
                      ExtraTreesClassifier, ExtraTreesRegressor]

    for Estimator in all_estimators:
        est = Estimator(min_impurity_decrease=0.1)
        est.fit(X, y)
        for tree in est.estimators_:
            # Simply check if the parameter is passed on correctly. Tree tests
            # will suffice for the actual working of this param
            assert tree.min_impurity_decrease == 0.1


@use_np
def test_forest_feature_importances_sum():
    X, y = datasets.make_classification(n_samples=15, n_informative=3, random_state=1,
                                        n_classes=3)
    X, y = np.array(X), np.array(y)
    clf = RandomForestClassifier(min_samples_leaf=5, random_state=42,
                                 n_estimators=200).fit(X, y)
    assert math.isclose(1, clf.feature_importances_.sum(), abs_tol=1e-7)


@use_np
def test_forest_degenerate_feature_importances():
    # build a forest of single node trees. See #13636
    X = np.zeros((10, 10))
    y = np.ones((10,))
    gbr = RandomForestRegressor(n_estimators=10).fit(X, y)
    assert_array_equal(gbr.feature_importances_,
                       np.zeros(10, dtype=np.float64))


@use_np
@pytest.mark.parametrize('name', FOREST_CLASSIFIERS_REGRESSORS)
@pytest.mark.parametrize(
    'max_samples, exc_type, exc_msg',
    [(int(1e9), ValueError,
      "`max_samples` must be in range 1 to 6 but got value 1000000000"),
     (1.0, ValueError,
      r"`max_samples` must be in range \(0, 1\) but got value 1.0"),
     (2.0, ValueError,
      r"`max_samples` must be in range \(0, 1\) but got value 2.0"),
     (0.0, ValueError,
      r"`max_samples` must be in range \(0, 1\) but got value 0.0"),
     (np.nan, ValueError,
      r"`max_samples` must be in range \(0, 1\) but got value nan"),
     (np.inf, ValueError,
      r"`max_samples` must be in range \(0, 1\) but got value inf"),
     ('str max_samples?!', TypeError,
      r"`max_samples` should be int or float, but got "
      r"type '\<class 'str'\>'"),
])
def test_max_samples_exceptions(name, max_samples, exc_type, exc_msg):
    # Check invalid `max_samples` values
    est = FOREST_CLASSIFIERS_REGRESSORS[name](max_samples=max_samples)
    with pytest.raises(exc_type, match=exc_msg):
        est.fit(X, y)


@use_np
@pytest.mark.parametrize(
    'ForestClass', [RandomForestClassifier, RandomForestRegressor]
)
def test_little_tree_with_small_max_samples(ForestClass):
    rng = _np.random.RandomState(1)

    X = rng.randn(10000, 2)
    y = rng.randn(10000) > 0

    # First fit with no restriction on max samples
    est1 = ForestClass(
        n_estimators=1,
        random_state=rng,
        max_samples=None,
    )

    # Second fit with max samples restricted to just 2
    est2 = ForestClass(
        n_estimators=1,
        random_state=rng,
        max_samples=2,
    )

    est1.fit(X, y)
    est2.fit(X, y)

    tree1 = est1.estimators_[0].tree_
    tree2 = est2.estimators_[0].tree_

    msg = "Tree without `max_samples` restriction should have more nodes"
    assert tree1.node_count > tree2.node_count, msg
