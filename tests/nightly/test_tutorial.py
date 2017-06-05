#pylint: disable=no-member, too-many-locals, too-many-branches, no-self-use, broad-except, lost-exception, too-many-nested-blocks, too-few-public-methods, invalid-name
"""
    This script converts all python tutorials into python script
    and tests whether there is any warning or error.
    After running python script, it will also convert markdown files
    to notebooks to make sure notebook execution has no error.
"""
import os
import warnings
import imp

import traceback
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

fail_dict = {}

def test_tutorial(file_path):
    """Run tutorial python script and  save any error or warning.
       If no error or warning occurs, run notebook.

    Parameters
    ----------
    file_path : str
        path of tutorial markdown file
    """
    with warnings.catch_warnings(record=True) as w:
        tutorial_name = os.path.basename(file_path)
        print file_path + '.py'
        try:
            imp.load_source('tutorial', file_path + '.py')
            if len(w) > 0:
                err_msg = "%s.py has %d warnings.\n" % (tutorial_name, len(w))
                fail_dict[tutorial_name] = err_msg
            else:
                test_tutorial_nb(file_path)
        except Exception:
            err_msg = "%s.py has error:\n%s" % (tutorial_name, traceback.format_exc())
            fail_dict[tutorial_name] = err_msg

def test_tutorial_nb(file_path):
    """Run tutorial jupyter notebook to catch any execution error.

    Parameters
    ----------
    file_path : str
        path of tutorial markdown file
    """
    tutorial_name = os.path.basename(file_path)
    notebook = nbformat.read(file_path + '_python.ipynb', as_version=4)
    eprocessor = ExecutePreprocessor(timeout=1800)
    try:
        eprocessor.preprocess(notebook, {'metadata': {}})
    except Exception as err:
        err_msg = "Python script successfully run without error or warning " \
                  "but notebook returned error:\n%s\nSomething weird happened." \
                  % (str(err))
        fail_dict[tutorial_name] = err_msg

if __name__ == "__main__":
    tutorial_dir = '../../docs/_build/html/tutorials/'
    with open('test_tutorial_config.txt') as config_file:
        tutorial_list = []
        for line in config_file:
            tutorial_list.append(line.lstrip().rstrip())
            file_dir = tutorial_dir + line.lstrip().rstrip()
            test_tutorial(file_dir)

        fail_num = len(fail_dict)
        success_num = len(tutorial_list) - fail_num
        print "Test Summary Start"
        print "%d tutorials tested:" % (len(tutorial_list))
        for tutorial in tutorial_list:
            print tutorial
        print "\n%d tests failed:" % (fail_num)
        for tutorial, msg in fail_dict.items():
            print tutorial + ":"
            print msg
        print "Test Summary End"
        print "Stats start"
        print "[Passed: %d of %d]" % (success_num, len(tutorial_list))
        print "Stats end"

