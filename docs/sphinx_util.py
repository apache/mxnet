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
    subprocess.call('cd ' + r_root +'; R -e "roxygen2::roxygenize()"; R CMD Rd2pdf . --no-preview -o ' + pdf_path, shell = True)
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

def convert_md_phase(phase):
    try:
        import pypandoc
    except:
        return phase
    return pypandoc.convert(phase, 'rst', format='md').replace('\n', ' ').replace('\r', '')

def build_table(table):
    if len(table) < 3:
        return ''
    out = '```eval_rst\n.. list-table::\n   :header-rows: 1\n\n'
    for i,l in enumerate(table):
        cols = l.split('|')[1:-1]
        if i == 0:
            ncol = len(cols)
        else:
            if len(cols) != ncol:
                return ''
        if i == 1:
            for c in cols:
                if len(c) is not 0 and '---' not in c:
                    return ''
        else:
            for j,c in enumerate(cols):
                if j == 0:
                    out += '   * - '
                else:
                    out += '     - '
                out += convert_md_phase(c)+ '\n'
    out += '```\n'
    return out

def convert_md_table(root_path):
    import glob
    import codecs
    files = []
    for i in range(5):
        files += glob.glob(os.path.join(root_path, *(['*']*i+['*.md'])))
    for f in files:
        started = False
        num_table = 0
        table = []
        output = ''
        with codecs.open(f, 'r', 'utf-8') as i:
            data = i.readlines()
        for l in data:
            r = l.strip()
            if r.startswith('|'):
                table += [r,]
                started = True
            else:
                if started is True:
                    tab = build_table(table)
                    if tab is not '':
                        num_table += 1
                        output += tab
                    started = False
                    table = []
                output += l
        if num_table != 0:
            print 'converted %d tables in %s' % (num_table, f)
            with codecs.open(f, 'w', 'utf-8') as i:
                i.write(output)

subprocess.call('./build-notebooks.sh')

curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
root_path = os.path.join(curr_path, '..')
convert_md_table(curr_path)
run_build_mxnet(root_path)
build_r_docs(root_path)
build_scala_docs(root_path)

if not os.path.exists('../recommonmark'):
    subprocess.call('cd ..; rm -rf recommonmark;' +
                    'git clone https://github.com/tqchen/recommonmark', shell = True)
else:
    subprocess.call('cd ../recommonmark/; git pull', shell=True)
sys.path.insert(0, os.path.abspath('../recommonmark/'))
from recommonmark import parser, transform
MarkdownParser = parser.CommonMarkParser
AutoStructify = transform.AutoStructify
