from enum import Enum
import re
import os
import sys
import subprocess
import traceback
import argparse

class Criteria:
    def __init__(self, metric_name, relation, value):
        self.metric_name = metric_name
        self.relation = relation
        self.value = float(value)
        
        self.lt = True if relation.lower() == "lt" else False
        self.gt = True if relation.lower() == "gt" else False

    def __str__(self):
        return "%s %s %s" % (self.metric_name, self.relation, self.value)

class Test:
    def __init__(self, train_val_prototxt, solver_prototxt, criterions):
        self.train_val_prototxt = train_val_prototxt
        self.solver_prototxt = solver_prototxt
        self.criterions = criterions

    def __str__(self):
        out  = "train_val_prototxt: %s\n" % self.train_val_prototxt
        out += "solver_prototxt: %s\n" % self.solver_prototxt
        # Apologies for using 'Criteria' as singular in the interest of readability.
        out += ("Criterions" if len(self.criterions)>1 else "Criteria") + ":\n"
        for criteria in self.criterions:
            out += "  %s\n" % criteria
        return out

def parse_test_description(test_desc_path):

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
                if (len(criterions) > 0) and (train_val_prototxt is not None) and (solver_prototxt is not None):

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
    if not os.path.exists(path):
        print("Creating directory %s" % path)
        os.makedirs(path)
    else:
        print("%s already exists" % path)

def translate(train_prototxt_path, solver_prototxt, out_dir_path):
	# Run caffetranslator --training-prototxt <train_val_prototxt_path> --solver <solver_prototxt_path> --output-file <output_file_path>
    print("Translating %s and %s. Writing output to %s" % (train_prototxt_path, solver_prototxt, out_dir_path))
    converted_file_path = out_dir_path + "/" + converted_file_name
    args = "--training-prototxt %s --solver %s --output-file %s" % (train_prototxt_path, solver_prototxt, converted_file_path)

    try:
        subprocess.run("caffetranslator " + args, shell=True, check=True, timeout=5)
    except:
        print("Failed to translate")
        traceback.print_exc(file=sys.stdout)
        return 1

    return 0

def train_network(out_dir_path):
    # Run python <output_dir_path>/<converted_file_name>
    print("Training translated network at %s" % out_dir_path)
    converted_file_path = out_dir_path + "/" + converted_file_name
    train_log_path = out_dir_path + "/" + train_log_name 
    command = "python " + converted_file_path + " 2>&1 | tee " + train_log_path
    try:
        subprocess.run(command, shell=True, check=True)
    except:
        print("Failed training the converted network")
        traceback.print_exc(file=sys.stdout)
        return 1

    return 0

def get_test_result(out_dir_path, criteria):
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

    result_str = "passed" if result == True else "failed"
    result_str = "%s: %f (%s)" % (criteria.metric_name, best_metric, result_str)

    return (result, result_str)

parser = argparse.ArgumentParser()
parser.add_argument("test_dir", help="Test directory containing the test description file. Check https://goo.gl/KazsLj.")
args = parser.parse_args()
working_dir = args.test_dir

output_dir = working_dir + "/output"
converted_file_name = "converted.py"
train_log_name = "train.log"
test_desc_path = working_dir + "/test_description.txt"

os.chdir(working_dir)

# Read tests description from the provided test description file
tests = parse_test_description(test_desc_path)

for test in tests:
    print(test)

# Create the output directory if it doesn't already exist
create_dir(output_dir)

# Open the report file
report_path = output_dir + "/" + "test_report.txt"
test_report = open(report_path, 'w')

out_dir_num = 0

for test in tests:

    # Create the output directory where the translated network and training logs will be saved
    out_dir_path = output_dir + "/" + str(out_dir_num) + "/"
    create_dir(out_dir_path)
    out_dir_num += 1

    # Translate the given network
    train_prototxt_path = test.train_val_prototxt
    solver_prototxt     = test.solver_prototxt
    translate(train_prototxt_path, solver_prototxt, out_dir_path)
    
    # Train the translated network
    train_network(out_dir_path)

    all_tests_passed = True

    # Observed metrics for this test (List of human readable string)
    observed_metrics = ""
    
    for criteria in test.criterions:
        # result is boolean denoting whether the observed metric satisfies the 
        # requirement mentioned in the test.
        # result_str is a descriptive human readable format of 'result'
        result, result_str = get_test_result(out_dir_path, criteria)
        observed_metrics += result_str + "\n"

        if result == False:
            all_tests_passed = False

    test_report.write("Test details:\n")
    test_report.write(str(test) + "\n")
    test_report.write("Observed metrics:\n")
    test_report.write(str(observed_metrics) + "\n")
    test_report.write("Result: " + ("Passed" if all_tests_passed else "Failed") + "\n\n")
    test_report.write("------------------------------------------------------------------\n\n")
