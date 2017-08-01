#!/usr/bin/env python

"""
Convert jupyter notebook into the markdown format. The notebook outputs will be
removed.

It is heavily adapted from https://gist.github.com/decabyte/0ed87372774cf5d34d7e
"""

import sys
import io
import os
import argparse
import nbformat


def remove_outputs(nb):
    """Removes the outputs cells for a jupyter notebook."""
    for cell in nb.cells:
        if cell.cell_type == 'code':
            cell.outputs = []


def clear_notebook(old_ipynb, new_ipynb):
    with io.open(old_ipynb, 'r') as f:
        nb = nbformat.read(f, nbformat.NO_CONVERT)

    remove_outputs(nb)

    with io.open(new_ipynb, 'w', encoding='utf8') as f:
        nbformat.write(nb, f, nbformat.NO_CONVERT)


def main():
    parser = argparse.ArgumentParser(
        description="Jupyter Notebooks to markdown"
    )

    parser.add_argument("notebook", nargs=1, help="The notebook to be converted.")
    parser.add_argument("-o", "--output", help="output markdown file")
    args = parser.parse_args()

    old_ipynb = args.notebook[0]
    new_ipynb = 'tmp.ipynb'
    md_file = args.output
    print md_file
    if not md_file:
        md_file = os.path.splitext(old_ipynb)[0] + '.md'


    clear_notebook(old_ipynb, new_ipynb)
    os.system('jupyter nbconvert ' + new_ipynb + ' --to markdown --output ' + md_file)
    with open(md_file, 'a') as f:
        f.write('<!-- INSERT SOURCE DOWNLOAD BUTTONS -->')
    os.system('rm ' + new_ipynb)

if __name__ == '__main__':
    main()
