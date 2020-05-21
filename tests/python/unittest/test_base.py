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
from mxnet.base import data_dir
from nose.tools import *
from mxnet.test_utils import environment
import os
import unittest
import logging
import os.path as op
import platform


def test_data_dir():
    prev_data_dir = data_dir()
    system = platform.system()
    # Test that data_dir() returns the proper default value when MXNET_HOME is not set
    if system != 'Windows':
        with environment('MXNET_HOME', None):
            assertEqual(data_dir(), op.join(op.expanduser('~'), '.mxnet'))
    # Test that data_dir() responds to an explicit setting of MXNET_HOME
    with environment('MXNET_HOME', '/tmp/mxnet_data'):
        assertEqual(data_dir(), '/tmp/mxnet_data')
    # Test that this test has not disturbed the MXNET_HOME value existing before the test
    assertEqual(data_dir(), prev_data_dir)
