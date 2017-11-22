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

# coding: utf-8
"""Run convergence tests on translated networks"""
import re
import os
import sys
import subprocess
import traceback
import argparse

converted_file_name = "converted.py"
train_log_name = "train.log"

class Criteria(object):
    def __init__(self, metric_name, relation, value):
        self.metric_name = metric_name
        self.relation = relation
        self.value = float(value)

        self.lt = True if relation.lower() == "lt" else False
        self.gt = True if relation.lower() == "gt" else False

    def __str__(self):
        return "%s %s %s" % (self.metric_name, self.relation, self.value)

class Test(object):
    def __init__(self, train_val_prototxt, solver_prototxt, criterions):
        self.train_val_prototxt = train_val_prototxt
        self.solver_prototxt = solver_prototxt
        self.criterions = criterions

    def __str__(self):
        out = "train_val_prototxt: %s\n" % self.train_val_prototxt
        out += "solver_prototxt: %s\n" % self.solver_prototxt
        # Apologies for using 'Criteria' as singular in the interest of readability.
        out += ("Criterions" if len(self.criterions) > 1 else "Criteria") + ":\n"
        for criteria in self.criterions:
            out += "  %s\n" % criteria
        return out

def parse_test_description(test_desc_path):
    """Parse the test description file"""

    with open(test_desc_path) as f:

        tests = []
        criterions = []
        train_val_prototxt = None
        solver_prototxt = None

        for line in f:

            if len(line) == 0 or line.isspace():
                continue

            # If this line is a criteria definition
            if re.match(r'\s', line):
                # Extract the criteria and append it to the criterions list
                line = line.strip()
                metric_name, relation, value = line.split()
                criteria = Criteria(metric_name, relation, value)
                criterions.append(criteria)

            else:
                # If we have finished seeing one test definition and starting to see the next one
                if (len(criterions) > 0) and (train_val_prototxt is not None)\
                        and (solver_prototxt is not None):

                    # Create a test with the information we have and append it to the tests list
                    test = Test(train_val_prototxt, solver_prototxt, criterions)
                    tests.append(test)

                    # reset criterions
                    criterions = []

                    # Read the new train_val and solver prototxt
                    train_val_prototxt, solver_prototxt = line.split()
                else:
                    # The first test definition
                    train_val_prototxt, solver_prototxt = line.split()

        # Process the last test definition
        test = Test(train_val_prototxt, solver_prototxt, criterions)
        tests.append(test)
    return tests

def create_dir(path):
    """Create a directory using the specified path (if one doesn't already exist)"""
    if not os.path.exists(path):
        print("Creating directory %s" % path)
        os.makedirs(path)
    else:
        print("%s already exists" % path)

def translate(train_prototxt_path, solver_prototxt, out_dir_path):
    """Run caffetranslator and translate the Caffe prototxt to MXNet Python code"""
    # Run caffetranslator --training-prototxt <train_val_prototxt_path>\
    #     --solver <solver_prototxt_path> --output-file <output_file_path>
    print("Translating %s and %s. Writing output to %s" % (train_prototxt_path, solver_prototxt,
                                                           out_dir_path))
    converted_file_path = out_dir_path + "/" + converted_file_name
    args = "--training-prototxt %s --solver %s --output-file %s" %\
           (train_prototxt_path, solver_prototxt, converted_file_path)

    try:
        subprocess.run("caffetranslator " + args, shell=True, check=True, timeout=5)
    except CalledProcessError:
        print("Failed to translate")
        traceback.print_exc(file=sys.stdout)
        return 1

    return 0

def train_network(out_dir_path):
    """Run training using the generated network"""
    # Run python <output_dir_path>/<converted_file_name>
    print("Training translated network at %s" % out_dir_path)
    converted_file_path = out_dir_path + "/" + converted_file_name
    train_log_path = out_dir_path + "/" + train_log_name
    command = "python " + converted_file_path + " 2>&1 | tee " + train_log_path
    try:
        subprocess.run(command, shell=True, check=True)
    except CalledProcessError:
        print("Failed training the converted network")
        traceback.print_exc(file=sys.stdout)
        return 1

    return 0

def get_test_result(out_dir_path, criteria):
    """Get result of a test based on the given criteria"""
    log_file_path = out_dir_path + "/" + train_log_name
    log_full_text = open(log_file_path).read()

    metric_name = criteria.metric_name
    search_pattern = r"'%s': \d+\.\d+" % metric_name
    metric_lines = re.findall(search_pattern, log_full_text)

    best_metric = sys.float_info.max if criteria.lt is True else sys.float_info.min

    for metric_line in metric_lines:
        metric = re.findall(r"\d+\.\d+", metric_line)
        metric = float(metric[0])

        if criteria.lt is True:
            best_metric = min(best_metric, metric)
        elif criteria.gt is True:
            best_metric = max(best_metric, metric)

    result = False
    if criteria.lt is True:
        result = True if best_metric <= criteria.value else False
    elif criteria.gt is True:
        result = True if best_metric >= criteria.value else False

    result_str = "passed" if result is True else "failed"
    result_str = "%s: %f (%s)" % (criteria.metric_name, best_metric, result_str)

    return (result, result_str)

def run_tests(working_dir):
    """Run tests using test.cfg in the provided working_dir as the test config file"""

    # Make working directory the current directory
    os.chdir(working_dir)

    output_dir = "output"
    test_desc_path = "test.cfg"

    # Read tests description from the provided test description file
    tests = parse_test_description(test_desc_path)

    # Create the output directory if it doesn't already exist
    create_dir(output_dir)

    # Open the report file
    report_path = output_dir + "/" + "test_report.txt"
    test_report = open(report_path, 'w')

    out_dir_num = 0

    num_tests_passed = 0
    num_tests_failed = 0

    for test in tests:

        # Create the output directory where the translated network and training logs will be saved
        out_dir_path = output_dir + "/" + str(out_dir_num) + "/"
        create_dir(out_dir_path)
        out_dir_num += 1

        # Translate the given network
        train_prototxt_path = test.train_val_prototxt
        solver_prototxt = test.solver_prototxt
        translate(train_prototxt_path, solver_prototxt, out_dir_path)

        # Train the translated network
        train_network(out_dir_path)

        all_criterions_passed = True

        # Observed metrics for this test (List of human readable string)
        observed_metrics = ""

        for criteria in test.criterions:
            # result is boolean denoting whether the observed metric satisfies the
            # requirement mentioned in the test.
            # result_str is a descriptive human readable format of 'result'
            result, result_str = get_test_result(out_dir_path, criteria)
            observed_metrics += result_str + "\n"

            if result is False:
                all_criterions_passed = False

        if all_criterions_passed:
            num_tests_passed += 1
        else:
            num_tests_failed += 1

        test_report.write("Test details:\n")
        test_report.write(str(test) + "\n")
        test_report.write("Observed metrics:\n")
        test_report.write(str(observed_metrics) + "\n")
        test_report.write("Result: " + ("Passed" if all_criterions_passed else "Failed") + "\n\n")
        test_report.write("------------------------------------------------------------------\n\n")

    test_report.write("Test summary: Total tests: %d, passed: %d, failed: %d\n\n" %
                      (len(tests), num_tests_passed, num_tests_failed))
    test_report.write("------------------------------------------------------------------\n\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("test_dir", help="Test directory containing the test description file."
                                         "Check https://goo.gl/NJ6X3N.")
    cl_args = parser.parse_args()
    run_tests(cl_args.test_dir)
