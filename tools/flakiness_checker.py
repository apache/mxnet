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


DEFAULT_NUM_TRIALS = 500

def run_test_trials(test_file, test_name, num_trials, seed):
    test_path = test_file + ":" + test_name
    
    new_env = os.environ
    new_env["MXNET_TEST_COUNT"] = str(num_trials)
    if seed is None:
        print("Using random seed")
    else:
        new_env["MXNET_TEST_SEED"] = seed

    code = subprocess.call(["nosetests", "-s","--verbose",test_path], env = new_env)
    print("nosetests completed with return code " + str(code))

def find_test_path(test_file):
    test_file += ".py"
    test_path = test_file
    top = str(subprocess.check_output(["git", "rev-parse", "--show-toplevel"]),
            errors= "strict").strip()

    for (path, names, files) in os.walk(top):
        if test_file in files:
            test_path = path + "/" + test_path
            return test_path
    raise FileNotFoundError("Could not find " + test_file)


class NameAction(argparse.Action):
    """Parses command line argument to get test file and test name"""
    def __call__(self, parser, namespace, values, option_string=None):
        name = values.split(".")
        setattr(namespace, "test_path", find_test_path(name[0]))
        setattr(namespace, "test_name", name[1])

def parse_args():
    parser = argparse.ArgumentParser(description="Check test for flakiness")
    
    parser.add_argument("test", action=NameAction,
                        help="file name and and function name of test, "
                        "provided in the format: file_name.test_name")
    
    parser.add_argument("-n", "--num-trials",
                        default=DEFAULT_NUM_TRIALS, type=int,
                        help="number of test trials, passed as "
                        "MXNET_TEST_COUNT, defaults to 500")

    parser.add_argument("-s", "--seed",
                        help="random seed, passed as MXNET_TEST_SEED")            
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    run_test_trials(args.test_path, args.test_name, args.num_trials, args.seed)
