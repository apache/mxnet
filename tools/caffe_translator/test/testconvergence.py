from enum import Enum
import re

class Criteria:
    def __init__(self, metric_name, relation, value):
        self.metric_name = metric_name
        self.relation = relation
        self.value = value

class Test:
    def __init__(self, train_val_prototxt, solver_prototxt, lst_criteria):
        self.train_val_prototxt = train_val_prototxt
        self.solver_prototxt = solver_prototxt
        self.lst_criteria = lst_criteria

    def __str__(self):
        out  = "train_val_prototxt: %s\n" % train_val_prototxt
        out += "solver_prototxt: %s\n" % solver_prototxt
        for criteria in lst_criteria:
            out += "  Criteria: %s %s %s\n" % (criteria.metric_name, criteria.relation, criteria.value)
        return out

input_dir = "CaffeModels"
test_desc_path = input_dir + "/test.txt"

tests = []
lst_criteria = []
train_val_prototxt = None
solver_prototxt = None

with open(test_desc_path) as f:
    
    for line in f:
        
        if len(line) == 0 or line.isspace():
            continue
        
        # If this line is a criteria definition
        if re.match(r'\s', line):
            # Extract the criteria and append it to the criterias list
            line = line.strip()
            metric_name, relation, value = line.split()
            criteria = Criteria(metric_name, relation, value)
            lst_criteria.append(criteria)

        else:
            # If we have finished seeing one test definition and starting to see the next one 
            if (len(lst_criteria) > 0) and (train_val_prototxt is not None) and (solver_prototxt is not None):

                # Create a test with the information we have and append it to the tests list
                test = Test(train_val_prototxt, solver_prototxt, lst_criteria)
                tests.append(test)
                
                # reset lst_criteria
                lst_criteria = []
            else:
                # The first test definition
                train_val_prototxt, solver_prototxt = line.split()
    
    # Process the last test definition
    test = Test(train_val_prototxt, solver_prototxt, lst_criteria)
    tests.append(test)


for test in tests:
    print(test)

