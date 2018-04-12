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
import shutil
import time
import argparse
import traceback
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import sys

fail_dict = {}
TIME_OUT = 1800

def test_tutorial_nb(file_path, workingdir, kernel=None):
    """Run tutorial jupyter notebook to catch any execution error.

    Parameters
    ----------
    file_path : str
        path of tutorial .ipynb file
    workingdir: str
        path of the directory to run the tutorial in
    kernel: str
        Default None
        name of the kernel to use, if none, will use first kernel 
        in the list
    """
    tutorial_name = os.path.basename(file_path)
    sys.stdout.write('Testing {}...'.format(file_path))
    sys.stdout.flush()
    tick = time.time()
    notebook = nbformat.read(file_path + '.ipynb', as_version=4)
    if kernel:
        eprocessor = ExecutePreprocessor(timeout=TIME_OUT, kernel_name=kernel)
    else:
        eprocessor = ExecutePreprocessor(timeout=TIME_OUT)
    success = True
    try:
        eprocessor.preprocess(notebook, {'metadata': {'path':workingdir}})
    except Exception as err:
        err_msg = str(err)
        fail_dict[tutorial_name] = err_msg
        success = False
    finally:
        output_file = os.path.join(workingdir, "output.txt")
        output_nb = open(output_file, mode='w')
        nbformat.write(notebook, output_nb)
        output_nb.close()
        output_nb = open(output_file, mode='r')
        for line in output_nb:
            if "Warning:" in line:
                success = False
                if tutorial_name in fail_dict:
                    fail_dict[tutorial_name] += "\n"+line
                else:
                    fail_dict[tutorial_name] = "Warning:\n"+line
        sys.stdout.write(' Elapsed time: {0:.2f}s '.format(time.time()-tick  ))
        sys.stdout.write(' [{}] \n'.format('Success' if success else 'Failed'))
        sys.stdout.flush()


if __name__ == "__main__":
    tutorial_dir = os.path.join('..','..','docs', '_build', 'html', 'tutorials')
    tick = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--tutorial", help="tutorial to test, if not set, read from test_tutorial_config.txt")
    parser.add_argument("--kernel", help="name of the jupyter kernel to use for the test")
    parser.add_argument("--no-cache", help="clean the temp directory", action="store_true", dest="no_cache")
    args = parser.parse_args()
    

    tutorial_list = []
    if args.tutorial:
        tutorial_list.append(args.tutorial)
    else:
        with open('test_tutorial_config.txt') as config_file:
            for line in config_file:
                tutorial_list.append(line.lstrip().rstrip())
    
    temp_dir = 'tmp_notebook'
    if args.no_cache:
        print("Cleaning and setting up temp directory '{}'".format(temp_dir))
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    kernel = args.kernel if args.kernel else None
    
    for tutorial in tutorial_list:
        file_dir = os.path.join(*([tutorial_dir]+tutorial.split('/')))
        working_dir = os.path.join(*([temp_dir]+tutorial.split('/')))
        if not os.path.isdir(working_dir):
            os.makedirs(working_dir)
        test_tutorial_nb(file_dir, working_dir, kernel)

    fail_num = len(fail_dict)
    success_num = len(tutorial_list) - fail_num
    print("Test Summary Start")
    print("%d tutorials tested:" % (len(tutorial_list)))
    for tutorial in tutorial_list:
        print(tutorial)
    print("\n%d tests failed:" % (fail_num))
    for tutorial, msg in fail_dict.items():
        print(tutorial + ":")
        print(msg)
    print("Test Summary End")
    print("Stats start")
    print("[Passed: %d of %d]" % (success_num, len(tutorial_list)))
    print("Total time: {:.2f}s".format(time.time()-tick))
    print("Stats end")

    if fail_num > 0:
        exit(1)

