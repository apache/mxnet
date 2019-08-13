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
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
from notebook_test import run_notebook


TUTORIAL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'docs', '_build', 'html', 'tutorials')
KERNEL = os.getenv('MXNET_TUTORIAL_TEST_KERNEL', None)
NO_CACHE = os.getenv('MXNET_TUTORIAL_TEST_NO_CACHE', False)

def _test_tutorial_nb(tutorial):
    """Run tutorial Jupyter notebook to catch any execution error.

    Parameters
    ----------
    tutorial : str
        the name of the tutorial to be tested

    Returns
    -------
        True if there are no warnings or errors.
    """
    return run_notebook(tutorial, TUTORIAL_DIR, kernel=KERNEL, no_cache=NO_CACHE)


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

def test_basic_reshape_transpose():
       assert _test_tutorial_nb('basic/reshape_transpose')

def test_gluon_customop():
    assert _test_tutorial_nb('gluon/customop')

def test_gluon_custom_layer():
    assert _test_tutorial_nb('gluon/custom_layer')

def test_gluon_transforms():
    assert _test_tutorial_nb('gluon/transforms')

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

def test_gluon_multi_gpu():
    assert _test_tutorial_nb('gluon/multi_gpu')

def test_gluon_save_load_params():
    assert _test_tutorial_nb('gluon/save_load_params')

def test_gluon_hybrid():
    assert _test_tutorial_nb('gluon/hybrid')
    
def test_gluon_pretrained_models():
    assert _test_tutorial_nb('gluon/pretrained_models')    

def test_gluon_learning_rate_finder():
    assert _test_tutorial_nb('gluon/learning_rate_finder')

def test_gluon_learning_rate_schedules():
    assert _test_tutorial_nb('gluon/learning_rate_schedules')

def test_gluon_learning_rate_schedules_advanced():
    assert _test_tutorial_nb('gluon/learning_rate_schedules_advanced')

def test_gluon_info_gan():
    assert _test_tutorial_nb('gluon/info_gan')

def test_nlp_cnn():
    assert _test_tutorial_nb('nlp/cnn')

def test_onnx_super_resolution():
    assert _test_tutorial_nb('onnx/super_resolution')

def test_onnx_export_mxnet_to_onnx():
    assert _test_tutorial_nb('onnx/export_mxnet_to_onnx')

def test_onnx_fine_tuning_gluon():
    assert _test_tutorial_nb('onnx/fine_tuning_gluon')

def test_onnx_inference_on_onnx_model():
    assert _test_tutorial_nb('onnx/inference_on_onnx_model')

def test_python_linear_regression():
    assert _test_tutorial_nb('python/linear-regression')

def test_python_logistic_regression() :
    assert _test_tutorial_nb('gluon/logistic_regression_explained')

def test_python_numpy_gotchas() :
    assert _test_tutorial_nb('gluon/gotchas_numpy_in_mxnet')

def test_gluon_end_to_end():
    assert _test_tutorial_nb('gluon/gluon_from_experiment_to_deployment')

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

def test_module_to_gluon():
    assert _test_tutorial_nb('python/module_to_gluon')

def test_python_types_of_data_augmentation():
    assert _test_tutorial_nb('python/types_of_data_augmentation')

def test_python_profiler():
    assert _test_tutorial_nb('python/profiler')

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

def test_vision_cnn_visualization():
    assert _test_tutorial_nb('vision/cnn_visualization')

def test_control_flow():
    assert _test_tutorial_nb('control_flow/ControlFlowTutorial')

def test_amp():
    assert _test_tutorial_nb('amp/amp_tutorial')
