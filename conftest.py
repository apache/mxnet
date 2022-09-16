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
"""conftest.py contains configuration for pytest.

Configuration file for tests in tests/ and scripts/ folders.

Note that fixtures of higher-scoped fixtures (such as ``session``) are
instantiated before lower-scoped fixtures (such as ``function``).

"""

import logging
import os
import random

import pytest


def pytest_configure(config):
    # Load the user's locale settings to verify that MXNet works correctly when the C locale is set
    # to anything other than the default value. Please see #16134 for an example of a bug caused by
    # incorrect handling of C locales.
    import locale
    locale.setlocale(locale.LC_ALL, "")


def pytest_sessionfinish(session, exitstatus):
    if exitstatus == 5:  # Don't fail if no tests were run
        session.exitstatus = 0


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Make test outcome available to fixture.

    https://docs.pytest.org/en/latest/example/simple.html#making-test-result-information-available-in-fixtures
    """
    # execute all other hooks to obtain the report object
    outcome = yield
    rep = outcome.get_result()

    # set a report attribute for each phase of a call, which can
    # be "setup", "call", "teardown"
    setattr(item, "rep_" + rep.when, rep)


@pytest.fixture(scope='module', autouse=True)
def module_scope_waitall(request):
    """A module scope fixture to issue waitall() operations between test modules."""
    yield

    try:
        import mxnet as mx
        mx.npx.waitall()
    except:
        # Use print() as module level fixture logging.warning messages never
        # shown to users. https://github.com/pytest-dev/pytest/issues/7819
        print('Unable to import numpy/mxnet. Skip mx.npx.waitall().')


@pytest.fixture(scope='module', autouse=True)
def module_scope_seed(request):
    """Module scope fixture to help reproduce test segfaults

    Sets and outputs rng seeds.

    The segfault-debug procedure on a module called test_module.py is:

    1. run "pytest --verbose test_module.py".  A seg-faulting output might be:

       [INFO] np, mx and python random seeds = 4018804151
       test_module.test1 ... ok
       test_module.test2 ... Illegal instruction (core dumped)

    2. Copy the module-starting seed into the next command, then run:

       MXNET_MODULE_SEED=4018804151 pytest --log-level=DEBUG --verbose test_module.py

       Output might be:

       [WARNING] **** module-level seed is set: all tests running deterministically ****
       [INFO] np, mx and python random seeds = 4018804151
       test_module.test1 ... [DEBUG] np and mx random seeds = 3935862516
       ok
       test_module.test2 ... [DEBUG] np and mx random seeds = 1435005594
       Illegal instruction (core dumped)

    3. Copy the segfaulting-test seed into the command:
       MXNET_TEST_SEED=1435005594 pytest --log-level=DEBUG --verbose test_module.py:test2
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
    if module_seed_str is None:
        seed = random.randint(0, 2**31-1)
    else:
        seed = int(module_seed_str)
        # Use print() as module level fixture logging.warning messages never
        # shown to users. https://github.com/pytest-dev/pytest/issues/7819
        print('*** module-level seed is set: all tests running deterministically ***')
    print('Setting module np/mx/python random seeds, '
          f'use MXNET_MODULE_SEED={seed} to reproduce.')
    old_state = random.getstate()
    random.seed(seed)
    try:
        import numpy as np
        import mxnet as mx
        np.random.seed(seed)
        mx.random.seed(seed)
    except:
        # Use print() as module level fixture logging.warning messages never
        # shown to users. https://github.com/pytest-dev/pytest/issues/7819
        print('Unable to import numpy/mxnet. Skip setting module-level seed.')

    # The MXNET_TEST_SEED environment variable will override MXNET_MODULE_SEED for tests with
    #  the 'with_seed()' decoration.  Inform the user of this once here at the module level.
    if os.getenv('MXNET_TEST_SEED') is not None:
        # Use print() as module level fixture logging.warning messages never
        # shown to users. https://github.com/pytest-dev/pytest/issues/7819
        print('*** test-level seed set: all "@with_seed()" tests run deterministically ***')

    yield  # run all tests in the module

    random.setstate(old_state)


@pytest.fixture(scope='function', autouse=True)
def function_scope_seed(request):
    """A function scope fixture that manages rng seeds.

    This fixture automatically initializes the python, numpy and mxnet random
    number generators randomly on every test run.

    def test_ok_with_random_data():
        ...

    To fix the seed used for a test case mark the test function with the
    desired seed:

    @pytest.mark.seed(1)
    def test_not_ok_with_random_data():
        '''This testcase actually works.'''
        assert 17 == random.randint(0, 100)

    When a test fails, the fixture outputs the seed used. The user can then set
    the environment variable MXNET_TEST_SEED to the value reported, then rerun
    the test with:

        pytest --verbose -s <test_module_name.py> -k <failing_test>

    To run a test repeatedly, install pytest-repeat and add the --count argument:

        pip install pytest-repeat
        pytest --verbose -s <test_module_name.py> -k <failing_test> --count 1000

    """

    seed = request.node.get_closest_marker('seed')
    env_seed_str = os.getenv('MXNET_TEST_SEED')

    if seed is not None:
        seed = seed.args[0]
        assert isinstance(seed, int)
    elif env_seed_str is not None:
        seed = int(env_seed_str)
    else:
        seed = random.randint(0, 2**31-1)
    old_state = random.getstate()
    random.seed(seed)
    try:
        import numpy as np
        import mxnet as mx
        np.random.seed(seed)
        mx.random.seed(seed)
    except:
        logging.warning('Unable to import numpy/mxnet. Skip setting function-level seed.')

    seed_message = f'Setting np/mx/python random seeds to {seed}. Use MXNET_TEST_SEED={seed} to reproduce.'

    # Always log seed on DEBUG log level. This makes sure we can find out the
    # value of the seed even if the test case causes a segfault and subsequent
    # teardown code is not run.
    logging.debug(seed_message)

    yield  # run the test

    if request.node.rep_setup.failed:
        logging.error("Setting up a test failed: {}", request.node.nodeid)
    elif request.node.rep_call.outcome == 'failed':
        # Either request.node.rep_setup.failed or request.node.rep_setup.passed should be True
        assert request.node.rep_setup.passed
        # On failure also log seed on WARNING log level
        error_message = f'Error seen with seeded test, use MXNET_TEST_SEED={seed} to reproduce'
        logging.warning(error_message)

    random.setstate(old_state)
