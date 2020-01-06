#!/usr/bin/env python

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


"""
Convert jupyter notebook into the markdown format. The notebook outputs will be
removed.

It is heavily adapted from https://gist.github.com/decabyte/0ed87372774cf5d34d7e
"""
from __future__ import print_function

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
    print(md_file)
    if not md_file:
        md_file = os.path.splitext(old_ipynb)[0] + '.md'


    clear_notebook(old_ipynb, new_ipynb)
    os.system('jupyter nbconvert ' + new_ipynb + ' --to markdown --output ' + md_file)
    with open(md_file, 'a') as f:
        f.write('<!-- INSERT SOURCE DOWNLOAD BUTTONS -->')
    os.system('rm ' + new_ipynb)

if __name__ == '__main__':
    main()
