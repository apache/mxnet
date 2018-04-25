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

#pylint: disable=no-member, too-many-locals, too-many-branches, no-self-use, broad-except, lost-exception, too-many-nested-blocks, too-few-public-methods, invalid-name
"""
    This file tests and ensures that all tutorials notebooks run
    without warning or exception.

    env variable MXNET_TUTORIAL_TEST_KERNEL controls which kernel to use
    when running the notebook. e.g:
    `export MXNET_TUTORIAL_TEST_KERNEL=python2`

    env variable MXNET_TUTORIAL_TEST_NO_CACHE controls whether to clean the
    temporary directory in which the notebook was run and re-download any
    resource file. The default behaviour is to not clean the directory. Set to '1'
    to force clean the directory. e.g:
    `export MXNET_TUTORIAL_TEST_NO_CACHE=1`
    NB: in the real CI, the tests will re-download everything since they start from
    a clean workspace.
"""
import os
import warnings
import imp
import shutil
import time
import argparse
import traceback
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import sys


# Maximum 10 minutes per test
# Reaching timeout causes test failure
TIME_OUT = 10*60
# Pin to ipython version 4
IPYTHON_VERSION = 4
temp_dir = 'tmp_notebook'

def _test_tutorial_nb(tutorial):
    """Run tutorial jupyter notebook to catch any execution error.

    Parameters
    ----------
    tutorial : str
        tutorial name in folder/tutorial format
    """

    tutorial_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'docs', '_build', 'html', 'tutorials')
    tutorial_path = os.path.join(*([tutorial_dir] + tutorial.split('/')))

    # see env variable docs in the doc string of the file
    kernel = os.getenv('MXNET_TUTORIAL_TEST_KERNEL', None)
    no_cache = os.getenv('MXNET_TUTORIAL_TEST_NO_CACHE', False)

    working_dir = os.path.join(*([temp_dir] + tutorial.split('/')))

    if no_cache == '1':
        print("Cleaning and setting up temp directory '{}'".format(working_dir))
        shutil.rmtree(temp_dir, ignore_errors=True)

    errors = []
    notebook = None
    if not os.path.isdir(working_dir):
        os.makedirs(working_dir)
    try:
        notebook = nbformat.read(tutorial_path + '.ipynb', as_version=IPYTHON_VERSION)
        if kernel is not None:
            eprocessor = ExecutePreprocessor(timeout=TIME_OUT, kernel_name=kernel)
        else:
            eprocessor = ExecutePreprocessor(timeout=TIME_OUT)
        nb, stuff = eprocessor.preprocess(notebook, {'metadata': {'path': working_dir}})
        print(stuff)
    except Exception as err:
        err_msg = str(err)
        errors.append(err_msg)
    finally:
        if notebook is not None:
            output_file = os.path.join(working_dir, "output.txt")
            nbformat.write(notebook, output_file)
            output_nb = open(output_file, mode='r')
            for line in output_nb:
                if "Warning:" in line:
                    errors.append("Warning:\n"+line)
        if len(errors) > 0:
            print('\n'.join(errors))
            return False
        return True



def test_basic_ndarray():
   assert _test_tutorial_nb('basic/ndarray')

def test_basic_ndarray_indexing():
    assert _test_tutorial_nb('basic/ndarray_indexing')

def test_basic_symbol():
    assert _test_tutorial_nb('basic/symbol')

def test_basic_module():
    assert _test_tutorial_nb('basic/module')

def test_basic_data():
    assert _test_tutorial_nb('basic/data')

def test_gluon_customop():
    assert _test_tutorial_nb('gluon/customop')

def test_gluon_custom_layer():
    assert _test_tutorial_nb('gluon/custom_layer')

def test_gluon_data_augmentation():
    assert _test_tutorial_nb('gluon/data_augmentation')

def test_gluon_datasets():
    assert _test_tutorial_nb('gluon/datasets')

def test_gluon_naming():
    assert _test_tutorial_nb('gluon/naming')

def test_gluon_ndarray():
    assert _test_tutorial_nb('gluon/ndarray')

def test_gluon_mnist():
    assert _test_tutorial_nb('gluon/mnist')

def test_gluon_autograd():
    assert _test_tutorial_nb('gluon/autograd')

def test_gluon_gluon():
    assert _test_tutorial_nb('gluon/gluon')

def test_gluon_hybrid():
    assert _test_tutorial_nb('gluon/hybrid')

def test_nlp_cnn():
    assert _test_tutorial_nb('nlp/cnn')

def test_onnx_super_resolution():
    assert _test_tutorial_nb('onnx/super_resolution')

def test_onnx_fine_tuning_gluon():
    assert _test_tutorial_nb('onnx/fine_tuning_gluon')

def test_onnx_inference_on_onnx_model():
    assert _test_tutorial_nb('onnx/inference_on_onnx_model')

def test_python_matrix_factorization():
    assert _test_tutorial_nb('python/matrix_factorization')

def test_python_linear_regression() :
    assert _test_tutorial_nb('python/linear-regression')

def test_python_mnist():
    assert _test_tutorial_nb('python/mnist')

def test_python_predict_image():
    assert _test_tutorial_nb('python/predict_image')

def test_python_data_augmentation():
    assert _test_tutorial_nb('python/data_augmentation')

def test_python_data_augmentation_with_masks():
    assert _test_tutorial_nb('python/data_augmentation_with_masks')

def test_python_kvstore():
    assert _test_tutorial_nb('python/kvstore')

def test_python_types_of_data_augmentation():
    assert _test_tutorial_nb('python/types_of_data_augmentation')

def test_sparse_row_sparse():
    assert _test_tutorial_nb('sparse/row_sparse')

def test_sparse_csr():
    assert _test_tutorial_nb('sparse/csr')

def test_sparse_train():
    assert _test_tutorial_nb('sparse/train')

def test_speech_recognition_ctc():
    assert _test_tutorial_nb('speech_recognition/ctc')

def test_unsupervised_learning_gan():
    assert _test_tutorial_nb('unsupervised_learning/gan')

def test_vision_large_scale_classification():
    assert _test_tutorial_nb('vision/large_scale_classification')
