# -*- coding: utf-8 -*-
"""Helper utilty function for customization."""
import sys
import os
import docutils
import subprocess

READTHEDOCS_BUILD = (os.environ.get('READTHEDOCS', None) is not None)


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

if not os.path.exists('../recommonmark'):
    subprocess.call('cd ..; rm -rf recommonmark;' +
                    'git clone https://github.com/tqchen/recommonmark', shell = True)
else:
    subprocess.call('cd ../recommonmark/; git pull', shell=True)

if not os.path.exists('web-data'):
    subprocess.call('rm -rf web-data;' +
                    'git clone https://github.com/dmlc/web-data', shell = True)
else:
    subprocess.call('cd web-data; git pull', shell=True)


run_build_mxnet("..")
sys.path.insert(0, os.path.abspath('../recommonmark/'))


sys.stderr.write('READTHEDOCS=%s\n' % (READTHEDOCS_BUILD))

from recommonmark import parser, transform

MarkdownParser = parser.CommonMarkParser
AutoStructify = transform.AutoStructify
