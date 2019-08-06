import sys
import os
import time
import notedown
import nbformat
from nbconvert.preprocessors.execute import ExecutePreprocessor
from nbconvert.preprocessors import CellExecutionError

def md2ipynb():
    assert len(sys.argv) == 3, 'usage: input.md output.rst'
    (src_fn, input_fn, output_fn) = sys.argv

    # timeout for each notebook, in sec
    timeout = 20 * 60
    # if enable evaluation
    do_eval = int(os.environ.get('EVAL', True))
    reader = notedown.MarkdownReader()
    with open(input_fn, 'r') as f:
        notebook = reader.read(f)
    notebook['metadata'].update({'language_info':{'name':'R'}}) # need to add language info for syntax highlight
    if do_eval:
        tic = time.time()
        executor = ExecutePreprocessor(timeout=timeout, kernel_name='ir')
        print('%s: Evaluated %s in %f sec'%(src_fn, input_fn, time.time()-tic))
        try:
            notebook, resources = executor.preprocess(notebook, resources={})
        except CellExecutionError:
            msg = 'Error executing the notebook "%s".\n\n' % input_fn
            msg += 'See notebook "%s" for the traceback.' % output_fn
            print(msg)
            raise
        finally:
            with open(output_fn, 'w') as f:
                f.write(nbformat.writes(notebook).encode('utf8'))
    print('%s: Write results into %s'%(src_fn, output_fn))

if __name__ == '__main__':
    md2ipynb()
