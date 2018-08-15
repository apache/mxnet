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
""" Checks the current branch for changes affecting tests

This script is used for automated flaky test detection, when changes
are detected that affect a test, this script will list all
affected affected tests in a file called tests.tmp, which will
be read by the check_tests.py script to check them for flakiness.
"""

import logging
import subprocess
import sys

import diff_collator
import dependency_analyzer

logger = logging.getLogger(__name__)
TEST_PREFIX = "test_"
TESTS_FILE = "tests.tmp"

def select_tests(changes):
    """returns tests that are dependent on given changes

    All python unit tests are top-level function with the prefix 
    "test_" in the function name. To get all tests, we simply 
    filter our changes by this prefix, stored in TEST_PREFIX.
    """
    top = subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
    top = top.decode("utf-8").splitlines()[0]
    deps = dependency_analyzer.find_dependents(changes, top)

    return [(filename, test) 
            for filename in deps.keys() 
            for test in deps[filename] 
            if test.startswith(TEST_PREFIX)]

def output_tests(tests):
    if not tests:
        return 1
    
    with open(TESTS_FILE, "w+") as f:
        for f, t in tests:
            f.write("{}:{}\n".format(f, t))
    
    return 0


if __name__ == "__main__":
    args = diff_collator.parse_args()
    try:
        logging.basicConfig(level=getattr(logging, args.level))
    except AttributeError:
        logging.basicConfig(level=logging.INFO)
        logger.warning("Invalid logging level: %s", args.level)

    diff_output = diff_collator.get_diff_output(args)
    changes = diff_collator.parser(diff_output)
    diff_collator.output_changes(changes)

    changes = {k:set(v.keys()) for k, v in  changes.items()}
    tests = select_tests(changes)
    sys.exit(output_tests(tests))
    