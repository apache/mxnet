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

import glob
import os
import re

# White list of tutorials that should not have a
# 'Download jupyter notebook' button or be added to the
# automated test suite.
# Rules to be in the whitelist:
# - not a python tutorial
whitelist = ['basic/index.md',
             'c++/basics.md',
             'c++/index.md',
             'c++/subgraphAPI.md',
             'c++/mxnet_cpp_inference_tutorial.md',
             'control_flow/index.md',
             'embedded/index.md',
             'embedded/wine_detector.md',
             'gluon/index.md',
             'mkldnn/index.md',
             'mkldnn/MKLDNN_README.md',
             'mkldnn/operator_list.md',
             'nlp/index.md',
             'onnx/index.md',
             'python/index.md',
             'r/CallbackFunction.md',
             'r/charRnnModel.md',
             'r/classifyRealImageWithPretrainedModel.md',
             'r/CustomIterator.md',
             'r/CustomLossFunction.md',
             'r/fiveMinutesNeuralNetwork.md',
             'r/index.md',
             'r/mnistCompetition.md',
             'r/MultidimLstm.md',
             'r/ndarray.md',
             'r/symbol.md',
             'scala/char_lstm.md',
             'scala/mnist.md',
             'scala/index.md',
             'scala/mxnet_scala_on_intellij.md',
             'scala/mxnet_java_install_and_run_examples.md',
             'sparse/index.md',
             'speech_recognition/index.md',
             'unsupervised_learning/index.md',
             'vision/index.md',
             'tensorrt/index.md',
             'tensorrt/inference_with_trt.md',
             'java/index.md',
             'java/mxnet_java_on_intellij.md',
             'java/ssd_inference.md',
             'amp/index.md']
whitelist_set = set(whitelist)

def test_tutorial_downloadable():
    """
    Make sure that every tutorial that isn't in the whitelist has the placeholder
    that enables notebook download
    """
    download_button_string = '<!-- INSERT SOURCE DOWNLOAD BUTTONS -->'

    tutorial_path = os.path.join(os.path.dirname(__file__), '..', '..', 'docs', 'tutorials')
    tutorials = glob.glob(os.path.join(tutorial_path, '**', '*.md'))

    for tutorial in tutorials:
        with open(tutorial, 'r') as file:
            lines= file.readlines()
        last = lines[-1]
        second_last = lines[-2]
        downloadable = download_button_string in last or download_button_string in second_last
        friendly_name = '/'.join(tutorial.split('/')[-2:])
        if not downloadable and friendly_name  not in whitelist_set:
            print(last, second_last)
            assert False, "{} is missing <!-- INSERT SOURCE DOWNLOAD BUTTONS --> as its last line".format(friendly_name)

def test_tutorial_tested():
    """
    Make sure that every tutorial that isn't in the whitelist
    has been added to the tutorial test file
    """
    tutorial_test_file = os.path.join(os.path.dirname(__file__), 'test_tutorials.py')
    f = open(tutorial_test_file, 'r')
    tutorial_test_text = '\n'.join(f.readlines())
    tutorial_path = os.path.join(os.path.dirname(__file__), '..', '..', 'docs', 'tutorials')
    tutorials = glob.glob(os.path.join(tutorial_path, '**', '*.md'))

    tested_tutorials = set(re.findall(r"assert _test_tutorial_nb\('(.*)'\)", tutorial_test_text))
    for tutorial in tutorials:
        friendly_name = '/'.join(tutorial.split('/')[-2:]).split('.')[0]
        if friendly_name not in tested_tutorials and friendly_name+".md" not in whitelist_set:
            assert False, "{} has not been added to the tests/tutorials/test_tutorials.py test_suite".format(friendly_name)
