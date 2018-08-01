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
import os
import unittest
import logging
import os.path as op
import platform

class MXNetDataDirTest(unittest.TestCase):
    def setUp(self):
        self.mxnet_data_dir = os.environ.get('MXNET_HOME')
        if 'MXNET_HOME' in os.environ:
            del os.environ['MXNET_HOME']

    def tearDown(self):
        if self.mxnet_data_dir:
            os.environ['MXNET_HOME'] = self.mxnet_data_dir
        else:
            if 'MXNET_HOME' in os.environ:
                del os.environ['MXNET_HOME']

    def test_data_dir(self,):
        prev_data_dir = data_dir()
        system = platform.system()
        if system != 'Windows':
            self.assertEqual(data_dir(), op.join(op.expanduser('~'), '.mxnet'))
        os.environ['MXNET_HOME'] = '/tmp/mxnet_data'
        self.assertEqual(data_dir(), '/tmp/mxnet_data')
        del os.environ['MXNET_HOME']
        self.assertEqual(data_dir(), prev_data_dir)


