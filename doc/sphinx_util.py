# -*- coding: utf-8 -*-
"""Helper utilty function for customization."""
import sys
import os
import docutils
import subprocess


def run_build_mxnet(folder):
    """Run the doxygen make command in the designated folder."""
    try:
        subprocess.call('cd ..; rm -rf dmlc-core;' +
                        'git clone https://github.com/dmlc/dmlc-core', shell = True)
        subprocess.call('cd ..; rm -rf mshadow;' +
                        'git clone https://github.com/dmlc/mshadow', shell = True)
        subprocess.call('cd ..; cp make/readthedocs.mk config.mk', shell = True)
        retcode = subprocess.call("cd %s; make" % folder, shell = True)
        if retcode < 0:
            sys.stderr.write("build terminated by signal %s" % (-retcode))
    except OSError as e:
        sys.stderr.write("build execution failed: %s" % e)

if os.environ.get('READTHEDOCS', None) == 'True':
    subprocess.call('cd ..; rm -rf recommonmark;' +
                    'git clone https://github.com/tqchen/recommonmark', shell = True)

run_build_mxnet("..")
sys.path.insert(0, os.path.abspath('../recommonmark/'))


from recommonmark import parser, transform

MarkdownParser = parser.CommonMarkParser
AutoStructify = transform.AutoStructify
