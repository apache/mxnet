import sys
import os
import time
import notedown
import nbformat

def md2ipynb():
    assert len(sys.argv) == 3, 'usage: input.md output.rst'
    (src_fn, input_fn, output_fn) = sys.argv

    # timeout for each notebook, in sec
    timeout = 20 * 60
    # if enable evaluation
    do_eval = int(os.environ.get('EVAL', True))

    reader = notedown.MarkdownReader(match='strict')
    with open(input_fn, 'r', encoding="utf8") as f:
        notebook = reader.read(f)
    if do_eval:
        tic = time.time()
        notedown.run(notebook, timeout)
        print('%s: Evaluated %s in %f sec'%(src_fn, input_fn, time.time()-tic))
    # need to add language info to for syntax highlight
    notebook['metadata'].update({'language_info':{'name':'python'}})
    with open(output_fn, 'w') as f:
        f.write(nbformat.writes(notebook))
    print('%s: Write results into %s'%(src_fn, output_fn))

if __name__ == '__main__':
    md2ipynb()
