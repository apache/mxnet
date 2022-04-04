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

import sys
from pathlib import Path
curr_path = Path(__file__).resolve().parent
sys.path.insert(0, str(curr_path.parent))

import mxnet as mx
import amp.common as amp_common_tests


AMP_DTYPE = 'bfloat16'


def test_amp_coverage():
    amp_common_tests.test_amp_coverage(AMP_DTYPE, 'BF16')


@mx.util.use_np
def test_amp_basic_use():
    amp_common_tests.test_amp_basic_use(AMP_DTYPE)


@mx.util.use_np
def test_amp_offline_casting():
    amp_common_tests.test_amp_offline_casting(AMP_DTYPE)


@mx.util.use_np
def test_amp_offline_casting_shared_params():
    amp_common_tests.test_amp_offline_casting_shared_params(AMP_DTYPE)


@mx.util.use_np
def test_lp16_fp32_ops_order_independence():
    amp_common_tests.test_lp16_fp32_ops_order_independence(AMP_DTYPE)
