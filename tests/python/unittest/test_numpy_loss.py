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

import mxnet as mx
import numpy as np
from mxnet import gluon, autograd
from mxnet.test_utils import assert_almost_equal, default_context, use_np
from common import xfail_when_nonstandard_decimal_separator
import pytest


@xfail_when_nonstandard_decimal_separator
@use_np
@pytest.mark.parametrize("hybridize", [False, True])
def test_loss_np_ndarray(hybridize):
    output = mx.np.array([1, 2, 3, 4])
    label = mx.np.array([1, 3, 5, 7])
    weighting = mx.np.array([0.5, 1, 0.5, 1])

    loss = gluon.loss.L1Loss()
    if hybridize:
        loss.hybridize()
    assert mx.np.sum(loss(output, label)) == 6.
    loss = gluon.loss.L1Loss(weight=0.5)
    if hybridize:
        loss.hybridize()
    assert mx.np.sum(loss(output, label)) == 3.
    loss = gluon.loss.L1Loss()
    if hybridize:
        loss.hybridize()
    assert mx.np.sum(loss(output, label, weighting)) == 5.

    loss = gluon.loss.L2Loss()
    if hybridize:
        loss.hybridize()
    assert mx.np.sum(loss(output, label)) == 7.
    loss = gluon.loss.L2Loss(weight=0.25)
    if hybridize:
        loss.hybridize()
    assert mx.np.sum(loss(output, label)) == 1.75
    loss = gluon.loss.L2Loss()
    if hybridize:
        loss.hybridize()
    assert mx.np.sum(loss(output, label, weighting)) == 6

    loss = gluon.loss.HuberLoss()
    if hybridize:
        loss.hybridize()
    assert mx.np.sum(loss(output, label)) == 4.5
    loss = gluon.loss.HuberLoss(weight=0.25)
    if hybridize:
        loss.hybridize()
    assert mx.np.sum(loss(output, label)) == 1.125
    loss = gluon.loss.HuberLoss()
    if hybridize:
        loss.hybridize()
    assert mx.np.sum(loss(output, label, weighting)) == 3.75

    loss = gluon.loss.HingeLoss(margin=10)
    if hybridize:
        loss.hybridize()
    assert mx.np.sum(loss(output, label)) == 13.
    loss = gluon.loss.HingeLoss(margin=8, weight=0.25)
    if hybridize:
        loss.hybridize()
    assert mx.np.sum(loss(output, label)) == 2.25
    loss = gluon.loss.HingeLoss(margin=7)
    if hybridize:
        loss.hybridize()
    assert mx.np.sum(loss(output, label, weighting)) == 4.

    loss = gluon.loss.SquaredHingeLoss(margin=10)
    if hybridize:
        loss.hybridize()
    assert mx.np.sum(loss(output, label)) == 97.
    loss = gluon.loss.SquaredHingeLoss(margin=8, weight=0.25)
    if hybridize:
        loss.hybridize()
    assert mx.np.sum(loss(output, label)) == 13.25
    loss = gluon.loss.SquaredHingeLoss(margin=7)
    if hybridize:
        loss.hybridize()
    assert mx.np.sum(loss(output, label, weighting)) == 19.

    loss = gluon.loss.TripletLoss(margin=10)
    if hybridize:
        loss.hybridize()
    assert mx.np.sum(loss(output, label, -label)) == 6.
    loss = gluon.loss.TripletLoss(margin=8, weight=0.25)
    if hybridize:
        loss.hybridize()
    assert mx.np.sum(loss(output, label, -label)) == 1.
    loss = gluon.loss.TripletLoss(margin=7)
    if hybridize:
        loss.hybridize()
    assert mx.np.sum(loss(output, label, -label, weighting)) == 1.5

    output = mx.np.array([[0, 2], [1, 4]])
    label = mx.np.array([0, 1])
    weighting = mx.np.array([[0.5], [1.0]])

    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    if hybridize:
        loss.hybridize()
    L = loss(output, label).asnumpy()
    assert_almost_equal(L, np.array([ 2.12692809,  0.04858733]), rtol=1e-3, atol=1e-4)

    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    if hybridize:
        loss.hybridize()
    L = loss(output, label, weighting).asnumpy()
    assert_almost_equal(L, np.array([ 1.06346405,  0.04858733]), rtol=1e-3, atol=1e-4)


@use_np
@pytest.mark.parametrize("hybridize", [False, True])
def test_bce_equal_ce2(hybridize):
    N = 100
    loss1 = gluon.loss.SigmoidBCELoss(from_sigmoid=True)
    if hybridize:
        loss1.hybridize()
    loss2 = gluon.loss.SoftmaxCELoss(from_logits=True)
    if hybridize:
        loss2.hybridize()
    out1 = mx.np.random.uniform(0.1, 0.9, size=(N, 1))
    out2 = mx.np.log(mx.np.concatenate((1-out1, out1), axis=1) + 1e-8)
    label = mx.np.round(mx.np.random.uniform(0, 1, size=(N, 1)))
    assert_almost_equal(loss1(out1, label).asnumpy(), loss2(out2, label).asnumpy())


@use_np
@pytest.mark.parametrize("hybridize", [False, True])
def test_logistic_loss_equal_bce(hybridize):
    N = 100
    loss_binary = gluon.loss.LogisticLoss(label_format='binary')
    if hybridize:
        loss_binary.hybridize()
    loss_signed = gluon.loss.LogisticLoss(label_format='signed')
    if hybridize:
        loss_signed.hybridize()
    loss_bce = gluon.loss.SigmoidBCELoss(from_sigmoid=False)
    if hybridize:
        loss_bce.hybridize()
    data = mx.np.random.uniform(-10, 10, size=(N, 1))
    label = mx.np.round(mx.np.random.uniform(0, 1, size=(N, 1)))
    assert_almost_equal(loss_binary(data, label), loss_bce(data, label), atol=1e-6)
    assert_almost_equal(loss_signed(data, 2 * label - 1), loss_bce(data, label), atol=1e-6)


@use_np
@pytest.mark.parametrize("hybridize", [False, True])
def test_ctc_loss(hybridize):
    loss = gluon.loss.CTCLoss()
    if hybridize:
        loss.hybridize()
    l = loss(mx.np.ones((2,20,4)), mx.np.array([[1,0,-1,-1],[2,1,1,-1]]))
    assert_almost_equal(l, np.array([18.82820702, 16.50581741]))

    loss = gluon.loss.CTCLoss(layout='TNC')
    if hybridize:
        loss.hybridize()
    l = loss(mx.np.ones((20,2,4)), mx.np.array([[1,0,-1,-1],[2,1,1,-1]]))
    assert_almost_equal(l, np.array([18.82820702, 16.50581741]))

    loss = gluon.loss.CTCLoss(layout='TNC', label_layout='TN')
    if hybridize:
        loss.hybridize()
    l = loss(mx.np.ones((20,2,4)), mx.np.array([[1,0,-1,-1],[2,1,1,-1]]).T)
    assert_almost_equal(l, np.array([18.82820702, 16.50581741]))

    loss = gluon.loss.CTCLoss()
    if hybridize:
        loss.hybridize()
    l = loss(mx.np.ones((2,20,4)), mx.np.array([[2,1,2,2],[3,2,2,2]]), None, mx.np.array([2,3]))
    assert_almost_equal(l, np.array([18.82820702, 16.50581741]))

    loss = gluon.loss.CTCLoss()
    if hybridize:
        loss.hybridize()
    l = loss(mx.np.ones((2,25,4)), mx.np.array([[2,1,-1,-1],[3,2,2,-1]]), mx.np.array([20,20]))
    assert_almost_equal(l, np.array([18.82820702, 16.50581741]))

    loss = gluon.loss.CTCLoss()
    if hybridize:
        loss.hybridize()
    l = loss(mx.np.ones((2,25,4)), mx.np.array([[2,1,3,3],[3,2,2,3]]), mx.np.array([20,20]), mx.np.array([2,3]))
    assert_almost_equal(l, np.array([18.82820702, 16.50581741]))


@xfail_when_nonstandard_decimal_separator
@use_np
def test_sdml_loss():

    N = 5 # number of samples
    DIM = 10 # Dimensionality
    EPOCHS = 20

    # Generate randomized data and 'positive' samples
    data = mx.np.random.uniform(-1, 1, size=(N, DIM))
    pos = data + mx.np.random.uniform(-0.1, 0.1, size=(N, DIM)) # correlated paired data
    data_iter = mx.io.NDArrayIter({'data' : data, 'pos' : pos}, batch_size=N)

    # Init model and trainer
    sdml_loss = gluon.loss.SDMLLoss()
    model = gluon.nn.Dense(DIM, activation='tanh') # Simple NN encoder
    model.initialize(mx.init.Xavier(), ctx=mx.current_context())
    trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate' : 0.1})

    for i in range(EPOCHS): # Training loop
        data_iter.reset()
        for iter_batch in data_iter:
            batch = [datum.as_in_ctx(mx.current_context()).as_np_ndarray() for datum in iter_batch.data]
            with autograd.record():
                data, pos = batch
                z_data, z_pos = model(data), model(pos)
                loss = sdml_loss(z_data, z_pos)
                loss.backward()
            trainer.step(1)

    # After training euclidean distance between aligned pairs should be lower than all non-aligned pairs
    avg_loss = loss.sum()/len(loss)
    assert(avg_loss < 0.05)


@use_np
@pytest.mark.parametrize("hybridize", [False, True])
def test_cosine_loss(hybridize):
    #Generating samples
    input1 = mx.np.random.randn(3, 2)
    input2 = mx.np.random.randn(3, 2)
    label = mx.np.sign(mx.np.random.randn(input1.shape[0]))
    #Calculating loss from cosine embedding loss function in Gluon
    Loss = gluon.loss.CosineEmbeddingLoss()
    if hybridize:
        Loss.hybridize()
    loss = Loss(input1, input2, label)

    # Calculating the loss Numpy way
    numerator = mx.np.sum(input1 * input2, keepdims=True, axis=1)
    denominator = mx.np.sqrt(mx.np.sum(input1**2, axis=1, keepdims=True)) \
    * mx.np.sqrt(mx.np.sum(input2**2, axis=1, keepdims=True))
    x = numerator/denominator
    label = mx.npx.reshape(label, (-1, 1))
    numpy_loss = mx.npx.reshape(
        mx.np.where(label == 1, 1-x, mx.npx.relu(x)), (-1,))
    assert_almost_equal(loss.asnumpy(), numpy_loss.asnumpy(), rtol=1e-3, atol=1e-5)


@xfail_when_nonstandard_decimal_separator
@use_np
@pytest.mark.parametrize("hybridize", [False, True])
def test_poisson_nllloss(hybridize):
    shape=(3, 4)
    not_axis0 = tuple(range(1, len(shape)))
    pred = mx.np.random.normal(size=shape)
    min_pred = mx.np.min(pred)
    #This is necessary to ensure only positive random values are generated for prediction,
    # to avoid ivalid log calculation
    pred[:] = pred + mx.np.abs(min_pred)
    target = mx.np.random.normal(size=shape)
    min_target = mx.np.min(target)
    #This is necessary to ensure only positive random values are generated for prediction,
    # to avoid ivalid log calculation
    target[:] += mx.np.abs(min_target)

    Loss = gluon.loss.PoissonNLLLoss(from_logits=True)
    if hybridize:
        Loss.hybridize()
    Loss_no_logits = gluon.loss.PoissonNLLLoss(from_logits=False)
    if hybridize:
        Loss_no_logits.hybridize()
    #Calculating by brute formula for default value of from_logits = True

    # 1) Testing for flag logits = True
    brute_loss = mx.np.mean(mx.np.exp(pred) - target * pred, axis=1)
    loss_withlogits = Loss(pred, target)
    assert_almost_equal(brute_loss, loss_withlogits)

    #2) Testing for flag logits = False
    loss_no_logits = Loss_no_logits(pred, target)
    np_loss_no_logits = mx.np.mean(pred - target * mx.np.log(pred + 1e-08),
                                   axis=1)
    assert_almost_equal(np_loss_no_logits, loss_no_logits)

    #3) Testing for Sterling approximation
    shape=(2, 3)
    np_pred = mx.np.random.uniform(1, 5, shape)
    np_target = mx.np.random.uniform(1, 5, shape)
    np_compute_full = mx.np.mean((np_pred - np_target * mx.np.log(np_pred + 1e-08)) + ((np_target * np.log(np_target)-\
     np_target + 0.5 * np.log(2 * np_target * np.pi))*(np_target > 1)), axis=1)
    Loss_compute_full = gluon.loss.PoissonNLLLoss(from_logits=False, compute_full=True)
    if hybridize:
        Loss_compute_full.hybridize()
    loss_compute_full = Loss_compute_full(np_pred, np_target)
    assert_almost_equal(np_compute_full, loss_compute_full)

