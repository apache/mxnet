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
import logging
import re
import os
import unittest
from straight_dope_test_utils import _test_notebook
from straight_dope_test_utils import _download_straight_dope_notebooks

NOTEBOOKS_WHITELIST = [
    'chapter01_crashcourse/preface',
    'chapter01_crashcourse/introduction',
    'chapter01_crashcourse/chapter-one-problem-set',
    'chapter02_supervised-learning/environment',
    'chapter03_deep-neural-networks/kaggle-gluon-kfold',
    'chapter04_convolutional-neural-networks/deep-cnns-alexnet',  # > 10 mins.
    'chapter06_optimization/gd-sgd-scratch',  # Overflow warning is intended.
    'chapter06_optimization/gd-sgd-gluon',  # Overflow warning is intended.
    'chapter07_distributed-learning/multiple-gpus-scratch',
    'chapter07_distributed-learning/multiple-gpus-gluon',
    'chapter07_distributed-learning/training-with-multiple-machines',
    'chapter11_recommender-systems/intro-recommender-systems',  # Early draft, non-working.
    'chapter12_time-series/intro-forecasting-gluon',
    'chapter12_time-series/intro-forecasting-2-gluon',
    'chapter13_unsupervised-learning/vae-gluon',
    'chapter18_variational-methods-and-uncertainty/bayes-by-backprop-rnn',
    'chapter17_deep-reinforcement-learning/DQN',
    'chapter17_deep-reinforcement-learning/DDQN',
    'chapter19_graph-neural-networks/Graph-Neural-Networks',
    'chapter16_tensor_methods/tensor_basics',
    'chapter18_variational-methods-and-uncertainty/bayes-by-backprop',  # > 10 mins.
    'chapter18_variational-methods-and-uncertainty/bayes-by-backprop-gluon',  # > 10 mins.
    'cheatsheets/kaggle-gluon-kfold'
]


class StraightDopeSingleGpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        logging.basicConfig(level=logging.INFO)
        assert _download_straight_dope_notebooks()

    def test_completeness(self):
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

    def test_ndarray(self):
        assert _test_notebook('chapter01_crashcourse/ndarray')

    def test_linear_algebra(self):
        assert _test_notebook('chapter01_crashcourse/linear-algebra')

    def test_probability(self):
        assert _test_notebook('chapter01_crashcourse/probability')

    def test_autograd(self):
        assert _test_notebook('chapter01_crashcourse/autograd')

    # Chapter 2

    def test_linear_regression_scratch(self):
        assert _test_notebook('chapter02_supervised-learning/linear-regression-scratch')

    def test_linear_regression_gluon(self):
        assert _test_notebook('chapter02_supervised-learning/linear-regression-gluon')

    def test_logistic_regression_gluon(self):
        assert _test_notebook('chapter02_supervised-learning/logistic-regression-gluon')

    def test_softmax_regression_scratch(self):
        assert _test_notebook('chapter02_supervised-learning/softmax-regression-scratch')

    def test_softmax_regression_gluon(self):
        assert _test_notebook('chapter02_supervised-learning/softmax-regression-gluon')

    def test_regularization_scratch(self):
        assert _test_notebook('chapter02_supervised-learning/regularization-scratch')

    def test_regularization_gluon(self):
        assert _test_notebook('chapter02_supervised-learning/regularization-gluon')

    def test_perceptron(self):
        assert _test_notebook('chapter02_supervised-learning/perceptron')

    # Chapter 3

    def test_mlp_scratch(self):
        assert _test_notebook('chapter03_deep-neural-networks/mlp-scratch')

    def test_mlp_gluon(self):
        assert _test_notebook('chapter03_deep-neural-networks/mlp-gluon')

    def test_mlp_dropout_scratch(self):
        assert _test_notebook('chapter03_deep-neural-networks/mlp-dropout-scratch')

    def test_mlp_dropout_gluon(self):
        assert _test_notebook('chapter03_deep-neural-networks/mlp-dropout-gluon')

    def test_plumbing(self):
        assert _test_notebook('chapter03_deep-neural-networks/plumbing')

    def test_custom_layer(self):
        assert _test_notebook('chapter03_deep-neural-networks/custom-layer')

    def test_serialization(self):
        assert _test_notebook('chapter03_deep-neural-networks/serialization')

    # Chapter 4

    def test_cnn_scratch(self):
        assert _test_notebook('chapter04_convolutional-neural-networks/cnn-scratch')

    def test_cnn_gluon(self):
        assert _test_notebook('chapter04_convolutional-neural-networks/cnn-gluon')

    def test_very_deep_nets_vgg(self):
        assert _test_notebook('chapter04_convolutional-neural-networks/very-deep-nets-vgg')

    def test_cnn_batch_norm_scratch(self):
        assert _test_notebook('chapter04_convolutional-neural-networks/cnn-batch-norm-scratch')

    def test_cnn_batch_norm_gluon(self):
        assert _test_notebook('chapter04_convolutional-neural-networks/cnn-batch-norm-gluon')

    # Chapter 5

    def test_simple_rnn(self):
        assert _test_notebook('chapter05_recurrent-neural-networks/simple-rnn')

    def test_lstm_scratch(self):
        assert _test_notebook('chapter05_recurrent-neural-networks/lstm-scratch')

    def test_gru_scratch(self):
        assert _test_notebook('chapter05_recurrent-neural-networks/gru-scratch')

    def test_rnn_gluon(self):
        assert _test_notebook('chapter05_recurrent-neural-networks/rnns-gluon')
 
    # Chapter 6

    def test_optimization_intro(self):
        assert _test_notebook('chapter06_optimization/optimization-intro')

    def test_momentum_scratch(self):
        assert _test_notebook('chapter06_optimization/momentum-scratch')

    def test_momentum_gluon(self):
        assert _test_notebook('chapter06_optimization/momentum-gluon')

    def test_adagrad_scratch(self):
        assert _test_notebook('chapter06_optimization/adagrad-scratch')

    def test_adagrad_gluon(self):
        assert _test_notebook('chapter06_optimization/adagrad-gluon')

    def test_rmsprop_scratch(self):
        assert _test_notebook('chapter06_optimization/rmsprop-scratch')

    def test_rmsprop_gluon(self):
        assert _test_notebook('chapter06_optimization/rmsprop-gluon')

    def test_adadelta_scratch(self):
        assert _test_notebook('chapter06_optimization/adadelta-scratch')

    def test_adadelta_gluon(self):
        assert _test_notebook('chapter06_optimization/adadelta-gluon')

    def test_adam_scratch(self):
        assert _test_notebook('chapter06_optimization/adam-scratch')

    def test_adam_gluon(self):
        assert _test_notebook('chapter06_optimization/adam-gluon')

    # Chapter 7

    def test_hybridize(self):
        assert _test_notebook('chapter07_distributed-learning/hybridize')


    # Chapter 8

    def test_object_detection(self):
        assert _test_notebook('chapter08_computer-vision/object-detection')

    def test_fine_tuning(self):
        assert _test_notebook('chapter08_computer-vision/fine-tuning')

    def test_visual_qa(self):
        assert _test_notebook('chapter08_computer-vision/visual-question-answer')


    # Chapter 9

    def test_tree_lstm(self):
        assert _test_notebook('chapter09_natural-language-processing/tree-lstm')

    # Chapter 12

    def test_lds_scratch(self):
        assert _test_notebook('chapter12_time-series/lds-scratch')

    def test_issm_scratch(self):
        assert _test_notebook('chapter12_time-series/issm-scratch')

    # Chapter 14

    def test_igan_intro(self):
        assert _test_notebook('chapter14_generative-adversarial-networks/gan-intro')

    def test_dcgan(self):
        assert _test_notebook('chapter14_generative-adversarial-networks/dcgan')

    def test_generative_adversarial_networks(self):
        assert _test_notebook('chapter14_generative-adversarial-networks/conditional')

    def test_pixel2pixel(self):
        assert _test_notebook('chapter14_generative-adversarial-networks/pixel2pixel')
