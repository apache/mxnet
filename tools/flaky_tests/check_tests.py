#!/usr/bin/env python3
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
""" Checks selected tests for flakiness

This script is used for automated flaky test detection. It reads the
tests listed in tests.tmp and runs them a large number of times.
Currently, each test is given an equal number of runs
such that all tests being checked can be run within the time budget.
"""
import logging
import time
import os
import sys

import flakiness_checker

LOGGING_FILE = os.path.join(os.path.dirname(__file__), "results.log")
TESTS_DIRECTORY = "tests/python"
TESTS_FILE = "tests.tmp"
PROFILING_TRIALS = 10
TIME_BUDGET = 5400 

logger = logging.getLogger(__name__)

fh = logging.FileHandler(LOGGING_FILE)
fh.setLevel(logging.INFO)
fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
logger.addHandler(fh)


def calculate_test_trials(tests):
    """Calculate the number of times each test should be run
    
    Currently, each test is run the same number of times, where the
    number is based on the time it takes to run each test once.
    """
    def time_test(test):
        start = time.time()
        flakiness_checker.run_test_trials(
            test[0], test[1], PROFILING_TRIALS + 1)
        end = time.time()
        profile_time = end - start

        start = time.time()
        flakiness_checker.run_test_trials(test[0], test[1], 1)
        end = time.time()
        setup_time = end - start

        actual_time = profile_time - setup_time
        return actual_time / PROFILING_TRIALS

    total_time = 0.0
    for t in tests:
        total_time += time_test(t)

    try:
        n = int(TIME_BUDGET / total_time)
    except ZeroDivisionError:
        logger.error("Total time for tests was 0")
        return []
    
    logger.debug("total_time: %f | num_trials: %d", total_time, n)
    return [(t, n) for t in tests]


def check_tests(tests):
    """Check given tests for flakiness"""
    flaky, nonflaky = [], []
    tests = calculate_test_trials(tests)

    for t, n in tests:
        res = flakiness_checker.run_test_trials(t[0], t[1], n)

        if res != 0:
            flaky.append(t)
        else:
            nonflaky.append(t)
    
    return flaky, nonflaky


def output_results(flaky, nonflaky):
    logger.info("Following tests failed flakiness checker:")
    if not flaky:
        logger.info("None")
    for test in flaky:
        logger.info("%s:%s", test[0], test[1])

    logger.info("Following tests passed flakiness checker:")
    if not nonflaky:
        logger.info("None")
    for test in nonflaky:
        logger.info("%s:%s", test[0], test[1])

    logger.info("[Results]\tTotal: %d\tFlaky: %d\tNon-flaky: %d",
                len(flaky) + len(nonflaky), len(flaky), len(nonflaky))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    tests = []
    with open(TESTS_FILE) as f:
        for line in f.readlines():
            test = line.split(":")
            tests.append((test[0], test[1]))
    
    os.remove(TESTS_FILE)
    
    flaky, nonflaky = check_tests(tests)
    output_results(flaky, nonflaky)

    if flaky:
        sys.exit(1)
