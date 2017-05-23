#pylint: disable=no-member, too-many-locals, too-many-branches, no-self-use, broad-except, lost-exception, too-many-nested-blocks, too-few-public-methods
"""
    This script converts all python tutorials into python script
    and tests whether there is any warning or error.
    After running python script, it will also convert markdown files
    to notebooks to make sure notebook execution has no error.
"""

import os
import warnings
import importlib
#pylint: enable=no-member

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

fail_dict = {}

def test_tutorial(file_path):
    with warnings.catch_warnings(record=True) as w:
        tutorial_name = os.path.basename(file_path)
        try:
            imp.load_source('module.name', file_path + '.py')
            if len(w) > 0:
                err_msg = "%s.py has %d warnings." % (tutorial_name, len(w))
                fail_dict[tutorial_name] = err_msg
            else:
                test_tutorial_nb(file_path)
        except Exception:
            err_msg = "%s.py has error:\n%s" % (tutorial_name, Exception)
            fail_dict[tutorial_name] = err_msg

def test_tutorial_nb(file_path):
    tutorial_name = os.path.basename(file_path)
    notebook = nbformat.read(file_path + '.ipynb', as_version=4)
    eprocessor = ExecutePreprocessor(timeout=900)
    try:
        eprocessor.preprocess(notebook, {'metadata': {}})
    except Exception as err:
        err_msg = "Python script successfully run without error or warning " \
                  "but notebook returned error:\n%s\nSomething weird happened." \
                  % (err)
        fail_dict[tutorial_name] = err_msg

if __name__ == "__main__":
    tutorial_dir = '../../docs/tutorials'
    with open('test_tutorial_config.txt') as config_file:
        tutorial_list = []
        for line in config_file:
            tutorial_list.append(line.lstrip().rstrip())
            file_path = tutorial_dir + line.lstrip().rstrip()
            test_tutorial(file_path)

        fail_num = len(fail_dict)
        success_num = len(tutorial_list) - fail_num
        print("Test Summary Start")
        print("%d tutorials tested:" % (len(tutorial_list)))
        for tutorial in tutorial_list:
            print(tutorial)
        print("\n%d tests failed:" % (fail_num))
        for tutorial, err_msg in fail_dict.items():
            print(tutorial + ":")
            print(err_msg)
        print("Test Summary End")
        print("Stats start")
        print("[Passed: %d of %d]" % (success_num, len(tutorial_list)))

