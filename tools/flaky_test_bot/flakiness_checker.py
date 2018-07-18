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

""" Checks a given test for flakiness
Takes the file name and function name of a test, as well as, optionally,
the number of trials to run and the random seed to use
"""

import subprocess
import sys
import os
import random
import argparse
import re
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_NUM_TRIALS = 10000

def run_test_trials(test_file, test_name, num_trials, seed=None):
    test_path = test_file + ":" + test_name
    logger.info("Testing-- %s", test_path)

    new_env = os.environ.copy()
    new_env["MXNET_TEST_COUNT"] = str(num_trials)
    
    if seed is None:
        logger.info("No test seed provided, using random seed")
    else:
        new_env["MXNET_TEST_SEED"] = str(seed)

    return  subprocess.call(["nosetests", test_path], env = new_env)

class NameAction(argparse.Action):
    """Parses command line argument to get test file and test name"""
    def find_test_path(test_file):
        """Searches for the test file and returns the path if found
        As a default, the currend working directory is the top of the search.
        If a directory was provided as part of the argument, the directory 
        will be joined with cwd unless it was an absolute path, in which case,
        the absolute path will be used instead. 
        """
        test_file += ".py"
        test_path = os.path.split(test_file)
        top = os.path.join(os.getcwd(), test_path[0])

        for (path, dirs, files) in os.walk(top):
            if test_path[1] in files:
                return  os.path.join(path, test_path[1])
        raise FileNotFoundError("Could not find " + test_path[1] + 
                                "in directory: " + top)

    def __call__(self, parser, namespace, values, option_string=None):
        name = re.split("\.py:|\.", values)
        if len(name) != 2:
            raise ValueError("Invalid argument format for test. Format: "
                             "<file-name>.<test-name> or"
                             " <directory>/<file>:<test-name>")
        setattr(namespace, "test_path", self.find_test_path(name[0]))
        setattr(namespace, "test_name", name[1])

def parse_args():
    parser = argparse.ArgumentParser(description="Check test for flakiness")
    
    parser.add_argument("test", action=NameAction,
                        help="file name and and function name of test, "
                        "provided in the format: <file-name>.<test-name> "
                        "or <directory>/<file>:<test-name>")
    
    parser.add_argument("-n", "--num-trials", metavar="N",
                        default=DEFAULT_NUM_TRIALS, type=int,
                        help="number of test trials, passed as "
                        "MXNET_TEST_COUNT, defaults to 500")

    parser.add_argument("-s", "--seed", type=int,
                        help="test seed, passed as MXNET_TEST_SEED, "
                        "defaults to random seed") 

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    run_test_trials(args.test_path, args.test_name, args.num_trials, args.seed)
    logger.info("Nosetests terminated with exit code %d", code)
