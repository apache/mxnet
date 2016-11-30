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

def build_r_docs(root_path):
    r_root = os.path.join(root_path, 'R-package')
    pdf_path = os.path.join(root_path, 'docs', 'api', 'r', 'mxnet-r-reference-manual.pdf')
    subprocess.call('cd ' + r_root +'; R CMD Rd2pdf . --no-preview -o ' + pdf_path, shell = True)
    dest_path = os.path.join(root_path, 'docs', '_build', 'html', 'api', 'r')
    subprocess.call('mkdir -p ' + dest_path, shell = True)
    subprocess.call('mv ' + pdf_path + ' ' + dest_path, shell = True)

def build_scala_docs(root_path):
    scala_path = os.path.join(root_path, 'scala-package', 'core', 'src', 'main', 'scala', 'ml', 'dmlc', 'mxnet')
    subprocess.call('cd ' + scala_path + '; scaladoc `find . | grep .*scala`', shell = True)

    dest_path = os.path.join(root_path, 'docs', '_build', 'html', 'api', 'scala', 'docs')
    subprocess.call('mkdir -p ' + dest_path, shell = True)

    scaladocs = ['index', 'index.html', 'ml', 'lib', 'index.js', 'package.html']
    for doc_file in scaladocs:
        subprocess.call('cd ' + scala_path + ';mv ' + doc_file + ' ' + dest_path, shell = True)

if not os.path.exists('../recommonmark'):
    subprocess.call('cd ..; rm -rf recommonmark;' +
                    'git clone https://github.com/tqchen/recommonmark', shell = True)
else:
    subprocess.call('cd ../recommonmark/; git pull', shell=True)

subprocess.call('./build-notebooks.sh')

curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
root_path = os.path.join(curr_path, '..')
run_build_mxnet(root_path)

build_r_docs(root_path)

build_scala_docs(root_path)

sys.path.insert(0, os.path.abspath('../recommonmark/'))

from recommonmark import parser, transform

MarkdownParser = parser.CommonMarkParser
AutoStructify = transform.AutoStructify
