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

#pylint: disable=no-member, too-many-locals, too-many-branches, no-self-use, broad-except, lost-exception, too-many-nested-blocks, too-few-public-methods, invalid-name, missing-docstring
"""
    This file tests that the notebooks requiring a single GPU run without
    warning or exception.
"""
import glob
import re
import os
from straight_dope_test_utils import _test_notebook

NOTEBOOKS_WHITELIST = [
    'chapter01_crashcourse/preface',
    'chapter01_crashcourse/introduction',
    'chapter01_crashcourse/chapter-one-problem-set',
    'chapter02_supervised-learning/environment',
]

def test_completeness():
    """
    Make sure that every tutorial that isn't in the whitelist is considered for testing by this
    file. Exceptions should be added to the whitelist.
    N.B. If the test is commented out, then that will be viewed as an intentional disabling of the
    test.
    """
    # Open up this test file.
    with open(__file__, 'r') as f:
        notebook_test_text = '\n'.join(f.readlines())

    notebooks_path = os.path.join(os.path.dirname(__file__), 'straight_dope_book')
    notebooks = glob.glob(os.path.join(notebooks_path, '**', '*.ipynb'))

    # Compile a list of notebooks that are tested
    tested_notebooks = set(re.findall(r"assert _test_notebook\('(.*)'\)", notebook_test_text))

   # Ensure each notebook in the straight dope book directory is on the whitelist or is tested.
    for notebook in notebooks:
        friendly_name = '/'.join(notebook.split('/')[-2:]).split('.')[0]
        if friendly_name not in tested_notebooks and friendly_name not in NOTEBOOKS_WHITELIST:
            assert False, friendly_name + " has not been added to the nightly/tests/straight_" + \
                          "dope/test_notebooks_single_gpu.py test_suite. Consider also adding " + \
                          "it to nightly/tests/straight_dope/test_notebooks_multi_gpu.py as " + \
                          "well if the notebooks makes use of multiple GPUs."

def test_ndarray():
    assert _test_notebook('chapter01_crashcourse/ndarray')

def test_linear_algebra():
    assert _test_notebook('chapter01_crashcourse/linear-algebra')

def test_probability():
    assert _test_notebook('chapter01_crashcourse/probability')

# TODO(vishaalk): Notebook contains the word 'Warning'. Needs to be updated to a synonym.
#def test_autograd():
#    assert _test_notebook('chapter01_crashcourse/autograd')

# Chapter 2

def test_linear_regression_scratch():
    assert _test_notebook('chapter02_supervised-learning/linear-regression-scratch')

def test_linear_regression_gluon():
    assert _test_notebook('chapter02_supervised-learning/linear-regression-gluon')

# TODO(vishaalk): There is a relative file path needs to be fixed so that the
# python code can be run from another directory.
#def test_logistic_regression_gluon():
#    assert _test_notebook('chapter02_supervised-learning/logistic-regression-gluon')

def test_softmax_regression_scratch():
    assert _test_notebook('chapter02_supervised-learning/softmax-regression-scratch')

def test_softmax_regression_gluon():
    assert _test_notebook('chapter02_supervised-learning/softmax-regression-gluon')

def test_regularization_scratch():
    assert _test_notebook('chapter02_supervised-learning/regularization-scratch')

# TODO(vishaalk): Notebook does not appear to be JSON: '{\n "cells": [\n  {\n   "cell_type": "m....
#def test_regularization_gluon():
#    assert _test_notebook('chapter02_supervised-learning/regularization-gluon')

def test_perceptron():
    assert _test_notebook('chapter02_supervised-learning/perceptron')

# Chapter 3

def test_mlp_scratch():
    assert _test_notebook('chapter03_deep-neural-networks/mlp-scratch')

def test_mlp_gluon():
    assert _test_notebook('chapter03_deep-neural-networks/mlp-gluon')

def test_mlp_dropout_scratch():
    assert _test_notebook('chapter03_deep-neural-networks/mlp-dropout-scratch')

def test_mlp_dropout_gluon():
    assert _test_notebook('chapter03_deep-neural-networks/mlp-dropout-gluon')

def test_plumbing():
    assert _test_notebook('chapter03_deep-neural-networks/plumbing')

def test_custom_layer():
    assert _test_notebook('chapter03_deep-neural-networks/custom-layer')

#def test_kaggle_gluon_kfold():
#    assert _test_notebook('chapter03_deep-neural-networks/kaggle-gluon-kfold')

# TODO(vishaalk): Load params and Save params are deprecated warning.
#def test_serialization():
#    assert _test_notebook('chapter03_deep-neural-networks/serialization')

# Chapter 4

def test_cnn_scratch():
    assert _test_notebook('chapter04_convolutional-neural-networks/cnn-scratch')

def test_cnn_gluon():
    assert _test_notebook('chapter04_convolutional-neural-networks/cnn-gluon')

# TODO(vishaalk): Load params and Save params are deprecated warning.
#def test_deep_cnns_alexnet():
#    assert _test_notebook('chapter04_convolutional-neural-networks/deep-cnns-alexnet')

def test_very_deep_nets_vgg():
    assert _test_notebook('chapter04_convolutional-neural-networks/very-deep-nets-vgg')

def test_cnn_batch_norm_scratch():
    assert _test_notebook('chapter04_convolutional-neural-networks/cnn-batch-norm-scratch')

def test_cnn_batch_norm_gluon():
    assert _test_notebook('chapter04_convolutional-neural-networks/cnn-batch-norm-gluon')

# Chapter 5

# TODO(vishaalk): There is a relative file path needs to be fixed so that the
# python code can be run from another directory.
#def test_simple_rnn():
#    assert _test_notebook('chapter05_recurrent-neural-networks/simple-rnn')

# TODO(vishaalk): There is a relative file path needs to be fixed so that the
# python code can be run from another directory.
#def test_lstm_scratch():
#    assert _test_notebook('chapter05_recurrent-neural-networks/lstm-scratch')

# TODO(vishaalk): There is a relative file path needs to be fixed so that the
# python code can be run from another directory.
#def test_gru_scratch():
#    assert _test_notebook('chapter05_recurrent-neural-networks/gru-scratch')

#def test_rnns_gluon():
#    assert _test_notebook('chapter05_recurrent-neural-networks/rnns-gluon')

# Chapter 6

def test_optimization_intro():
    assert _test_notebook('chapter06_optimization/optimization-intro')

# TODO(vishaalk): RuntimeWarning: Overflow encountered in reduce.
#def test_gd_sgd_scratch():
#    assert _test_notebook('chapter06_optimization/gd-sgd-scratch')

# TODO(vishaalk): RuntimeWarning: Overflow encountered in reduce.
#def test_gd_sgd_gluon():
#    assert _test_notebook('chapter06_optimization/gd-sgd-gluon')

def test_momentum_scratch():
    assert _test_notebook('chapter06_optimization/momentum-scratch')

def test_momentum_gluon():
    assert _test_notebook('chapter06_optimization/momentum-gluon')

def test_adagrad_scratch():
    assert _test_notebook('chapter06_optimization/adagrad-scratch')

def test_adagrad_gluon():
    assert _test_notebook('chapter06_optimization/adagrad-gluon')

def test_rmsprop_scratch():
    assert _test_notebook('chapter06_optimization/rmsprop-scratch')

def test_rmsprop_gluon():
    assert _test_notebook('chapter06_optimization/rmsprop-gluon')

def test_adadelta_scratch():
    assert _test_notebook('chapter06_optimization/adadelta-scratch')

def test_adadelta_gluon():
    assert _test_notebook('chapter06_optimization/adadelta-gluon')

def test_adam_scratch():
    assert _test_notebook('chapter06_optimization/adam-scratch')

def test_adam_gluon():
    assert _test_notebook('chapter06_optimization/adam-gluon')

# Chapter 7

def test_hybridize():
    assert _test_notebook('chapter07_distributed-learning/hybridize')

# TODO(vishaalk): module 'mxnet.gluon' has no attribute 'autograd'
#def test_multiple_gpus_scratch():
#    assert _test_notebook('chapter07_distributed-learning/multiple-gpus-scratch')

def test_multiple_gpus_gluon():
    assert _test_notebook('chapter07_distributed-learning/multiple-gpus-gluon')

def test_training_with_multiple_machines():
    assert _test_notebook('chapter07_distributed-learning/training-with-multiple-machines')

# Chapter 8

# TODO(vishaalk): Load params and Save params are deprecated warning.
#def test_object_detection():
#    assert _test_notebook('chapter08_computer-vision/object-detection')

# TODO(vishaalk): Module skimage needs to be added to docker image.
#def test_fine_tuning():
#    assert _test_notebook('chapter08_computer-vision/fine-tuning')

# TODO(vishaalk):
#def test_visual_question_answer():
#    assert _test_notebook('chapter08_computer-vision/visual-question-answer')

# Chapter 9

def test_tree_lstm():
    assert _test_notebook('chapter09_natural-language-processing/tree-lstm')

# Chapter 11

# TODO(vishaalk): Deferred initialization failed because shape cannot be inferred.
#def test_intro_recommender_systems():
#    assert _test_notebook('chapter11_recommender-systems/intro-recommender-systems')

# Chapter 12

def test_lds_scratch():
    assert _test_notebook('chapter12_time-series/lds-scratch')

# TODO(vishaalk): File doesn't appear to be valid JSON.
#def test_issm_scratch():
#    assert _test_notebook('chapter12_time-series/issm-scratch')

# TODO(vishaalk): Error: sequential1_batchnorm0_running_mean' has not been initialized
# def test_intro_forecasting_gluon():
#    assert _test_notebook('chapter12_time-series/intro-forecasting-gluon')

#def test_intro_forecasting_2_gluon():
#    assert _test_notebook('chapter12_time-series/intro-forecasting-2-gluon')

# Chapter 13

# TODO(vishaalk): Load params and Save params are deprecated warning.
#def test_vae_gluon():
#    assert _test_notebook('chapter13_unsupervised-learning/vae-gluon')

# Chapter 14

def test_igan_intro():
    assert _test_notebook('chapter14_generative-adversarial-networks/gan-intro')

def test_dcgan():
    assert _test_notebook('chapter14_generative-adversarial-networks/dcgan')

def test_generative_adversarial_networks():
    assert _test_notebook('chapter14_generative-adversarial-networks/conditional')

# Chapter 16

# TODO(vishaalk): Checked failed oshape.Size() != dshape.Size()
#def test_tensor_basics():
#    assert _test_notebook('chapter16_tensor_methods/tensor_basics')

# TODO(vishaalk): Notebook does not appear to be valid JSON.
#def test_pixel2pixel():
#    assert _test_notebook('chapter14_generative-adversarial-networks/pixel2pixel')

# Chapter 17

# TODO(vishaalk): Requires OpenAI Gym. Also uses deprecated load_params.
#def test_dqn():
#    assert _test_notebook('chapter17_deep-reinforcement-learning/DQN')

#def test_ddqn():
#    assert _test_notebook('chapter17_deep-reinforcement-learning/DDQN')

# Chapter 18

#def test_bayes_by_backprop():
#    assert _test_notebook('chapter18_variational-methods-and-uncertainty/bayes-by-backprop')

#def test_bayes_by_backprop_gluon():
#    assert _test_notebook('chapter18_variational-methods-and-uncertainty/bayes-by-backprop-gluon')

# TODO(vishaalk): AttributeError: 'list' object has no attribute 'keys'
#def test_bayes_by_backprop_rnn():
#    assert _test_notebook('chapter18_variational-methods-and-uncertainty/bayes-by-backprop-rnn')

# Chapter 19

# TODO(vishaalk): Requires deepchem
#def test_graph_neural_networks():
#    assert _test_notebook('chapter19_graph-neural-networks/Graph-Neural-Networks')

# Cheatsheets

# TODO(vishaalk): There is a relative file path needs to be fixed so that the
# python code can be run from another directory.
#def test_kaggle_gluon_kfold():
#    assert _test_notebook('cheatsheets/kaggle-gluon-kfold')
