"""
    This script runs notebooks in selected directory and report
    errors for each notebook.
    
    Traceback information can be found in the output notebooks
    generated in coresponding output directories.
    
    Before running this scripe, make sure all the notebooks have
    been run at least once and outputs are generated.
"""

import os
import errno
import json
import ConfigParser
import re
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import nbformat
from textwrap import dedent
import nbconvert.preprocessors.execute as execute

time_limit_flag = '# @@@ AUTOTEST_TIME_LIMT_SECONDS='
ignored_cell_flag = '# @@@ AUTOTEST_OUTPUT_IGNORED_CELL'

class CustomizedPreprocessor(execute.ExecutePreprocessor):
    """A customized preprocessor which allows preset for cell.
    In this test script, timeout is set before executing a cell.
    """
    def preprocess_cell(self, cell, resources, cell_index):
        """
        Executes a code cell with timeout. Default timeout is 900 sec.
        """
        if cell.cell_type != 'code':
            return cell, resources

        regex = re.compile(time_limit_flag + '[0-9]+')
        time_flag = re.search(regex, cell.source)
        if time_flag is not None:
            timeout = int(re.search(r'[0-9]+', time_flag).group())
            self.timeout = timeout

        outputs = self.run_cell(cell)
        cell.outputs = outputs

        if not self.allow_errors:
            for out in outputs:
                if out.output_type == 'error':
                    pattern = u"""\
                        An error occurred while executing cell No.{cell.execution_count}:
                        ------------------
                        {cell.source}
                        ------------------
                        {out.ename}: {out.evalue}
                        """
                    msg = dedent(pattern).format(out=out, cell=cell)
                    raise execute.CellExecutionError(msg)
        return cell, resources


class NotebookTester(object):
    """The class of notebook automated testing. A NotebookTester loads a test_config
    file and execute each notebook. A report containing detail traceback information
    will be generated.
    """
    def __init__(self, test_config):
        self.test_config = test_config

    def __read_config(self, test_config):
        """Read notebooks to be tested from test config file.

        Parameters
        ----------
        test_config : str
        test configuration file

        Returns
        -------
        nb_list : list
        Notebook list to be tested
        """
        nb_list = []
        configParser = ConfigParser.RawConfigParser()
        configParser.read(test_config)
        test_dirs = configParser.get('Folder Path', 'test_path').split(', ')
        if len(test_dirs) == 1 and len(test_dirs[0]) == 0:
            test_dirs.append('.')
        ignored_item = configParser.get('Folder Path', 'test_ignored').split(', ')
        ignored_dir = set()
        ignored_nb = set()
        for item in ignored_item:
            if item == '@@@ IGNORE_ALL':
                return nb_list
            if item.endswith('.ipynb'):
                ignored_nb.add(os.path.abspath(item))
            else:
                for root, _, _ in os.walk(item):
                    ignored_dir.add(os.path.abspath(root))
        for dir in test_dirs:
            for root, dirs, files in os.walk(dir):
                if os.path.abspath(root) in ignored_dir:
                    continue
                for file in files:
                    if file.endswith('.ipynb') and not file.endswith('-checkpoint.ipynb'):
                        notebook = os.path.join(root, file)
                        if os.path.abspath(notebook) not in ignored_nb:
                            if notebook.startswith('./'):
                                notebook = notebook[2:]
                            nb_list.append(notebook)
        return nb_list
            
        
    def __notebook_run(self, path):
        """Execute a notebook via nbconvert and collect output.
        
        Parameters
        ----------
        path : str
        notebook file path.
        
        Returns
        -------
        error : str
        notebook first cell execution errors.
        """
        error = ""
        parent_dir, nb_name = os.path.split(path)        
        with open(path) as nb_file:
            nb = nbformat.read(nb_file, as_version=4)
            ep = CustomizedPreprocessor(timeout=900)
            #Use a loop to avoid "Kernel died before replying to kernel_info" error, repeat 5 times
            for _ in range(0, 5):
                error = ""
                try:
                    ep.preprocess(nb, {'metadata': {'path': parent_dir}})
                except Exception as e:
                    error = str(e)
                finally:
                    if error != 'Kernel died before replying to kernel_info':
                        output_nb = os.path.splitext(nb_name)[0] + "_output.ipynb"
                        with open(output_nb, mode='w') as f:
                            nbformat.write(nb, f)
                        f.close()
                        nb_file.close()
                        if len(error) == 0:
                            cell_num = self.__verify_output(path, output_nb)
                            if cell_num > 0:
                                error = "Output in cell No.%d has changed." % cell_num
                        os.remove(output_nb)
                        return error
        return error


    def __verify_output(self, origin_nb, output_nb):
        """Compare the output cells of testing output notebook with original notebook.

        Parameters
        ----------
        origin_nb : str
        original notebook file path.
        
        output_nb : str
        output notebook file path.
        
        Returns
        -------
        cell_num : int
        First cell number in which outputs are incompatible
        """
        cell_num = 0
        origin_nb_file = open(origin_nb)
        origin_nb_js = json.load(origin_nb_file)
        output_nb_file = open(output_nb)
        output_nb_js = json.load(output_nb_file)
        for origin_cell, output_cell in zip(origin_nb_js["cells"], output_nb_js["cells"]):
            is_ignored_cell = False
            if len(origin_cell["source"]) == 0 or not origin_cell.has_key("outputs"):
                is_ignored_cell = True
            for line in origin_cell["source"]:
                if line.startswith(ignored_cell_flag):
                    is_ignored_cell = True
                    break
            if is_ignored_cell:
                continue
            if self.__extract_output(origin_cell["outputs"]) != self.__extract_output(output_cell["outputs"]):
                cell_num = origin_cell["execution_count"]
                break
        origin_nb_file.close()
        output_nb_file.close()
        return cell_num


    def __extract_output(self, outputs):
        """Extract text part of output of a notebook cell.
        
        Parasmeters
        -----------
        outputs : list
        list of output
        
        Returns
        -------
        ret : str
        Concatenation of all text output contents
        """
        ret = ''
        for dict in outputs:
            for key, val in dict.items():
                if str(key).startswith('text'):
                    for content in val:
                        ret += str(content)
                elif key == 'data':
                    for dt_key, dt_val in val.items():
                        if str(dt_key).startswith('text') and not str(dt_key).startswith('text/html'):
                            for dt_content in dt_val:
                                if not str(dt_content).startswith('<matplotlib') and not \
                                   str(dt_content).startswith('<graphviz'):
                                    ret += str(dt_content)
        return ret


    def run_test(self):
        """Run test using config file
        """
        nb_to_test = self.__read_config(self.test_config)
        test_summary = open('test_summary.txt', mode='w')
        fail_nb_dict = {}
        test_summary.write("%d notebooks were tested:\n" % len(nb_to_test))
        for nb in nb_to_test:
            test_summary.write("%s\n" % nb)
            print "Start to test %s.\n" % nb
            error = self.__notebook_run(nb)
            if len(error) == 0:
                print "Tests for %s all passed!\n" % nb
            else:
                fail_nb_dict[nb] = error
                print "Tests for %s failed:\n" % nb
                print error + '\n'
                if (error == 'Cell execution timed out, see log for details.' or 
                    error == 'Kernel died before replying to kernel_info'):
                    print "Please manually run this notebook to debug.\n"
        print "%d notebooks tested, %d succeeded, %d failed" % (len(nb_to_test),
                                                                len(nb_to_test) - len(fail_nb_dict),
                                                                len(fail_nb_dict))
        if len(fail_nb_dict) > 0:
            test_summary.write("\n%d notebook tests failed:\n" % len(fail_nb_dict))
            print "Following are failed notebooks:"
            for nb, error in fail_nb_dict.items():
                test_summary.write("\n%s:\n" % nb)
                test_summary.write("%s\n" % error)                       
                print nb
        else:
            test_summary.write("All notebook tests passed!")
        test_summary.close()
        print "Test summarys are stored in test_summary.txt"

if __name__ == "__main__":
    nb_tester = NotebookTester('test_config.txt')
    nb_tester.run_test()