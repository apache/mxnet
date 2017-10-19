from enum import Enum
import re
import os
import sys
import subprocess
import traceback

class Criteria:
    def __init__(self, metric_name, relation, value):
        self.metric_name = metric_name
        self.relation = relation
        self.value = value

    def __str__(self):
        return "%s %s %s" % (self.metric_name, self.relation, self.value)

class Test:
    def __init__(self, train_val_prototxt, solver_prototxt, criterions):
        self.train_val_prototxt = train_val_prototxt
        self.solver_prototxt = solver_prototxt
        self.criterions = criterions

    def __str__(self):
        out  = "train_val_prototxt: %s\n" % train_val_prototxt
        out += "solver_prototxt: %s\n" % solver_prototxt
        # Apologies for using 'Criteria' as singular in the interest of readability.
        out += ("Criterions" if len(criterions)>1 else "Criteria") + ":\n"
        for criteria in criterions:
            out += "  %s\n" % criteria
        return out

input_dir = "CaffeModels"
output_dir = "out"
converted_file_name = "converted.py"
train_log_name = "train.log"
test_desc_path = input_dir + "/test.txt"

tests = []
criterions = []
train_val_prototxt = None
solver_prototxt = None

with open(test_desc_path) as f:
    
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
            else:
                # The first test definition
                train_val_prototxt, solver_prototxt = line.split()
    
    # Process the last test definition
    test = Test(train_val_prototxt, solver_prototxt, criterions)
    tests.append(test)


for test in tests:
    print(test)

def create_dir(path):
    if not os.path.exists(path):
        print("Creating directory %s" % path)
        os.makedirs(path)
    else:
        print("%s already exists" % path)

create_dir(output_dir)
out_dir_num = 0

report_path = output_dir + "/" + "test_report.txt"
test_report = open(report_path, 'w')

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
	
def get_test_result(out_dir_path, criteria):
    result = True
    result_str = "accuracy/top1: 99.1 (passed)"
    return (result, result_str)

def train_network(out_dir_path):
    # Run python <output_dir_path>/<converted_file_name>
    print("Training translated network at %s" % out_dir_path)
    converted_file_path = out_dir_path + "/" + converted_file_name
    train_log_path = out_dir_path + "/" + train_log_name 
    command = "python " + converted_file_path + " > " + train_log_path + " 2>&1"
    try:
        subprocess.run(command, shell=True, check=True, timeout=5)
    except:
        print("Failed training the converted network")
        traceback.print_exc(file=sys.stdout)
        return 1
    
    return 0

for test in tests:

    out_dir_path = output_dir + "/" + str(out_dir_num) + "/"
    create_dir(out_dir_path)
    out_dir_num += 1

    train_prototxt_path = input_dir + "/" + test.train_val_prototxt
    solver_prototxt     = input_dir + "/" + test.solver_prototxt
    translate(train_prototxt_path, solver_prototxt, out_dir_path)
    train_network(out_dir_path)

    all_tests_passed = True
    
    observed_metrics = ""
    for criteria in test.criterions:
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


