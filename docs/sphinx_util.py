# -*- coding: utf-8 -*-
"""Helper utilty function for customization."""
import sys
import os
import docutils
import subprocess

def run_build_mxnet(folder):
    """Run the doxygen make command in the designated folder."""
    try:
        subprocess.call('cd %s; cp make/readthedocs.mk config.mk' % folder, shell = True)
        subprocess.call('cd %s; rm -rf build' % folder, shell = True)
        retcode = subprocess.call("cd %s; make -j$(nproc)" % folder, shell = True)
        if retcode < 0:
            sys.stderr.write("build terminated by signal %s" % (-retcode))
    except OSError as e:
        sys.stderr.write("build execution failed: %s" % e)

if not os.path.exists('../recommonmark'):
    subprocess.call('cd ..; rm -rf recommonmark;' +
                    'git clone https://github.com/tqchen/recommonmark', shell = True)
else:
    subprocess.call('cd ../recommonmark/; git pull', shell=True)

curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
root_path = os.path.join(curr_path, '..')
run_build_mxnet(root_path)

sys.path.insert(0, os.path.abspath('../recommonmark/'))

from recommonmark import parser, transform

MarkdownParser = parser.CommonMarkParser
AutoStructify = transform.AutoStructify
