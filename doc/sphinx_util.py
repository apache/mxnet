# -*- coding: utf-8 -*-
"""Helper utilty function for customization."""
import sys
import os
import docutils
import subprocess


#READTHEDOCS_BUILD = (os.environ.get('READTHEDOCS', None) == 'True')
READTHEDOCS_BUILD = True

def run_build_mxnet(folder):
    """Run the doxygen make command in the designated folder."""
    try:
        if READTHEDOCS_BUILD:
            subprocess.call('cd ..; cp make/readthedocs.mk config.mk', shell = True)
            subprocess.call('cd ..; rm -rf build', shell = True)
        retcode = subprocess.call("cd %s; make" % folder, shell = True)
        if retcode < 0:
            sys.stderr.write("build terminated by signal %s" % (-retcode))
    except OSError as e:
        sys.stderr.write("build execution failed: %s" % e)

if READTHEDOCS_BUILD or not os.path.exists('../recommonmark'):
    subprocess.call('cd ..; rm -rf recommonmark;' +
                    'git clone https://github.com/tqchen/recommonmark', shell = True)

run_build_mxnet("..")
sys.path.insert(0, os.path.abspath('../recommonmark/'))


from recommonmark import parser, transform

MarkdownParser = parser.CommonMarkParser
AutoStructify = transform.AutoStructify
