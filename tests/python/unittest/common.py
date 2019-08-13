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

from __future__ import print_function
import sys, os, logging
import multiprocessing as mp
import mxnet as mx
import numpy as np
import random
import shutil
from mxnet.base import MXNetError
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.append(os.path.join(curr_path, '../common/'))
sys.path.insert(0, os.path.join(curr_path, '../../../python'))

import models
from contextlib import contextmanager
from nose.tools import make_decorator, assert_raises
import tempfile

def assertRaises(expected_exception, func, *args, **kwargs):
    try:
        func(*args, **kwargs)
    except expected_exception as e:
        pass
    else:
        # Did not raise exception
        assert False, "%s did not raise %s" % (func.__name__, expected_exception.__name__)


def default_logger():
    """A logger used to output seed information to nosetests logs."""
    logger = logging.getLogger(__name__)
    # getLogger() lookups will return the same logger, but only add the handler once.
    if not len(logger.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
        logger.addHandler(handler)
        if (logger.getEffectiveLevel() == logging.NOTSET):
            logger.setLevel(logging.INFO)
    return logger


@contextmanager
def random_seed(seed=None):
    """
    Runs a code block with a new seed for np, mx and python's random.

    Parameters
    ----------

    seed : the seed to pass to np.random, mx.random and python's random.

    To impose rng determinism, invoke e.g. as in:

    with random_seed(1234):
        ...

    To impose rng non-determinism, invoke as in:

    with random_seed():
        ...

    Upon conclusion of the block, the rng's are returned to
    a state that is a function of their pre-block state, so
    any prior non-determinism is preserved.

    """

    try:
        next_seed = np.random.randint(0, np.iinfo(np.int32).max)
        if seed is None:
            np.random.seed()
            seed = np.random.randint(0, np.iinfo(np.int32).max)
        logger = default_logger()
        logger.debug('Setting np, mx and python random seeds = %s', seed)
        np.random.seed(seed)
        mx.random.seed(seed)
        random.seed(seed)
        yield
    finally:
        # Reinstate prior state of np.random and other generators
        np.random.seed(next_seed)
        mx.random.seed(next_seed)
        random.seed(next_seed)


def assert_raises_cudnn_not_satisfied(min_version):
    def test_helper(orig_test):
        @make_decorator(orig_test)
        def test_new(*args, **kwargs):
            cudnn_off = os.getenv('CUDNN_OFF_TEST_ONLY') == 'true'
            cudnn_env_version = os.getenv('CUDNN_VERSION', None if cudnn_off else '7.3.1')
            cudnn_test_disabled = cudnn_off or cudnn_env_version < min_version
            if not cudnn_test_disabled or mx.context.current_context().device_type == 'cpu':
                orig_test(*args, **kwargs)
            else:
                assert_raises((MXNetError, RuntimeError), orig_test, *args, **kwargs)
        return test_new
    return test_helper


def with_seed(seed=None):
    """
    A decorator for nosetests test functions that manages rng seeds.

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

        nosetests --verbose -s <test_module_name.py>:<failing_test>

    To run a test repeatedly, set MXNET_TEST_COUNT=<NNN> in the environment.
    To see the seeds of even the passing tests, add '--logging-level=DEBUG' to nosetests.
    """
    def test_helper(orig_test):
        @make_decorator(orig_test)
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
                logger = default_logger()
                # 'nosetests --logging-level=DEBUG' shows this msg even with an ensuing core dump.
                test_count_msg = '{} of {}: '.format(i+1,test_count) if test_count > 1 else ''
                test_msg = ('{}Setting test np/mx/python random seeds, use MXNET_TEST_SEED={}'
                            ' to reproduce.').format(test_count_msg, this_test_seed)
                logger.log(log_level, test_msg)
                try:
                    orig_test(*args, **kwargs)
                except:
                    # With exceptions, repeat test_msg at INFO level to be sure it's seen.
                    if log_level < logging.INFO:
                        logger.info(test_msg)
                    raise
                finally:
                    np.random.set_state(post_test_state)
        return test_new
    return test_helper


def setup_module():
    """
    A function with a 'magic name' executed automatically before each nosetests module
    (file of tests) that helps reproduce a test segfault by setting and outputting the rng seeds.

    The segfault-debug procedure on a module called test_module.py is:

    1. run "nosetests --verbose test_module.py".  A seg-faulting output might be:

       [INFO] np, mx and python random seeds = 4018804151
       test_module.test1 ... ok
       test_module.test2 ... Illegal instruction (core dumped)

    2. Copy the module-starting seed into the next command, then run:

       MXNET_MODULE_SEED=4018804151 nosetests --logging-level=DEBUG --verbose test_module.py

       Output might be:

       [WARNING] **** module-level seed is set: all tests running deterministically ****
       [INFO] np, mx and python random seeds = 4018804151
       test_module.test1 ... [DEBUG] np and mx random seeds = 3935862516
       ok
       test_module.test2 ... [DEBUG] np and mx random seeds = 1435005594
       Illegal instruction (core dumped)

    3. Copy the segfaulting-test seed into the command:
       MXNET_TEST_SEED=1435005594 nosetests --logging-level=DEBUG --verbose test_module.py:test2
       Output might be:

       [INFO] np, mx and python random seeds = 2481884723
       test_module.test2 ... [DEBUG] np and mx random seeds = 1435005594
       Illegal instruction (core dumped)

    3. Finally reproduce the segfault directly under gdb (might need additional os packages)
       by editing the bottom of test_module.py to be

       if __name__ == '__main__':
           logging.getLogger().setLevel(logging.DEBUG)
           test2()

       MXNET_TEST_SEED=1435005594 gdb -ex r --args python test_module.py

    4. When finished debugging the segfault, remember to unset any exported MXNET_ seed
       variables in the environment to return to non-deterministic testing (a good thing).
    """

    module_seed_str = os.getenv('MXNET_MODULE_SEED')
    logger = default_logger()
    if module_seed_str is None:
        seed = np.random.randint(0, np.iinfo(np.int32).max)
    else:
        seed = int(module_seed_str)
        logger.warn('*** module-level seed is set: all tests running deterministically ***')
    logger.info('Setting module np/mx/python random seeds, use MXNET_MODULE_SEED=%s to reproduce.', seed)
    np.random.seed(seed)
    mx.random.seed(seed)
    random.seed(seed)
    # The MXNET_TEST_SEED environment variable will override MXNET_MODULE_SEED for tests with
    #  the 'with_seed()' decoration.  Inform the user of this once here at the module level.
    if os.getenv('MXNET_TEST_SEED') is not None:
        logger.warn('*** test-level seed set: all "@with_seed()" tests run deterministically ***')

try:
    from tempfile import TemporaryDirectory
except:
    # really simple implementation of TemporaryDirectory
    class TemporaryDirectory(object):
        def __init__(self, suffix='', prefix='', dir=''):
            self._dirname = tempfile.mkdtemp(suffix, prefix, dir)

        def __enter__(self):
            return self._dirname

        def __exit__(self, exc_type, exc_value, traceback):
            shutil.rmtree(self._dirname)

def teardown():
    """
    A function with a 'magic name' executed automatically after each nosetests test module.

    It waits for all operations in one file to finish before carrying on the next.
    """
    mx.nd.waitall()


def run_in_spawned_process(func, env, *args):
    """
    Helper function to run a test in its own process.

    Avoids issues with Singleton- or otherwise-cached environment variable lookups in the backend.
    Adds a seed as first arg to propagate determinism.

    Parameters
    ----------

    func : function to run in a spawned process.
    env : dict of additional environment values to set temporarily in the environment before exec.
    args : args to pass to the function.

    Returns
    -------
    Whether the python version supports running the function as a spawned process.

    This routine calculates a random seed and passes it into the test as a first argument.  If the
    test uses random values, it should include an outer 'with random_seed(seed):'.  If the
    test needs to return values to the caller, consider use of shared variable arguments.
    """
    try:
        mpctx = mp.get_context('spawn')
    except:
        print('SKIP: python%s.%s lacks the required process fork-exec support ... ' %
              sys.version_info[0:2], file=sys.stderr, end='')
        return False
    else:
        seed = np.random.randint(0,1024*1024*1024)
        orig_environ = os.environ.copy()
        try:
            for (key, value) in env.items():
                os.environ[key] = str(value)
            # Prepend seed as first arg
            p = mpctx.Process(target=func, args=(seed,)+args)
            p.start()
            p.join()
            assert p.exitcode == 0, "Non-zero exit code %d from %s()." % (p.exitcode, func.__name__)
        finally:
            os.environ.clear()
            os.environ.update(orig_environ)
    return True