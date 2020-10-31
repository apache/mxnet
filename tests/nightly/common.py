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

import functools
import logging
import os
import random

import mxnet as mx
import numpy as np


def with_seed(seed=None):
    """
    A decorator for test functions that manages rng seeds.

    Parameters
    ----------

    seed : the seed to pass to np.random and mx.random


    This tests decorator sets the np, mx and python random seeds identically
    prior to each test, then outputs those seeds if the test fails or
    if the test requires a fixed seed (as a reminder to make the test
    more robust against random data).

    @with_seed()
    def test_ok_with_random_data():
        ...

    @with_seed(1234)
    def test_not_ok_with_random_data():
        ...

    Use of the @with_seed() decorator for all tests creates
    tests isolation and reproducability of failures.  When a
    test fails, the decorator outputs the seed used.  The user
    can then set the environment variable MXNET_TEST_SEED to
    the value reported, then rerun the test with:

        pytest --verbose --capture=no <test_module_name.py>::<failing_test>

    To run a test repeatedly, set MXNET_TEST_COUNT=<NNN> in the environment.
    To see the seeds of even the passing tests, add '--log-level=DEBUG' to pytest.
    """
    def test_helper(orig_test):
        @functools.wraps(orig_test)
        def test_new(*args, **kwargs):
            test_count = int(os.getenv('MXNET_TEST_COUNT', '1'))
            env_seed_str = os.getenv('MXNET_TEST_SEED')
            for i in range(test_count):
                if seed is not None:
                    this_test_seed = seed
                    log_level = logging.INFO
                elif env_seed_str is not None:
                    this_test_seed = int(env_seed_str)
                    log_level = logging.INFO
                else:
                    this_test_seed = np.random.randint(0, np.iinfo(np.int32).max)
                    log_level = logging.DEBUG
                post_test_state = np.random.get_state()
                np.random.seed(this_test_seed)
                mx.random.seed(this_test_seed)
                random.seed(this_test_seed)
                # 'pytest --logging-level=DEBUG' shows this msg even with an ensuing core dump.
                test_count_msg = '{} of {}: '.format(i+1,test_count) if test_count > 1 else ''
                pre_test_msg = ('{}Setting test np/mx/python random seeds, use MXNET_TEST_SEED={}'
                                ' to reproduce.').format(test_count_msg, this_test_seed)
                on_err_test_msg = ('{}Error seen with seeded test, use MXNET_TEST_SEED={}'
                                ' to reproduce.').format(test_count_msg, this_test_seed)
                logging.log(log_level, pre_test_msg)
                try:
                    orig_test(*args, **kwargs)
                except:
                    # With exceptions, repeat test_msg at WARNING level to be sure it's seen.
                    if log_level < logging.WARNING:
                        logging.warning(on_err_test_msg)
                    raise
                finally:
                    # Provide test-isolation for any test having this decorator
                    mx.nd.waitall()
                    np.random.set_state(post_test_state)
        return test_new
    return test_helper

