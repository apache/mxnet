"""A sphnix-doc plugin to build mxnet docs"""
import subprocess
import re
import os
import json
from recommonmark import transform
import pypandoc

# start or end of a code block
_CODE_MARK = re.compile('^([ ]*)```([\w]*)')

# language names and the according file extensions and comment symbol
_LANGS = {'python' : ('py', '#'),
          'r' : ('R','#'),
          'scala' : ('scala', '#'),
          'julia' : ('jl', '#'),
          'perl' : ('pl', '#'),
          'cpp' : ('cc', '//'),
          'bash' : ('sh', '#')}

_LANG_SELECTION_MARK = 'INSERT SELECTION BUTTONS'
_SRC_DOWNLOAD_MARK = 'INSERT SOURCE DOWNLOAD BUTTONS'

def _run_cmd(cmds):
    """Run commands, raise exception if failed"""
    if not isinstance(cmds, str):
        cmds = "".join(cmds)
    print("Execute \"%s\"" % cmds)
    try:
        subprocess.check_call(cmds, shell=True)
    except subprocess.CalledProcessError as err:
        print(err)
        raise err

def generate_doxygen(app):
    """Run the doxygen make commands"""
    _run_cmd("cd %s/.. && make doxygen" % app.builder.srcdir)
    _run_cmd("cp -rf doxygen/html %s/doxygen" % app.builder.outdir)

def build_mxnet(app):
    """Build mxnet .so lib"""
    _run_cmd("cd %s/.. && cp make/config.mk config.mk && make -j$(nproc) DEBUG=1" %
            app.builder.srcdir)

def build_r_docs(app):
    """build r pdf"""
    r_root = app.builder.srcdir + '/../R-package'
    pdf_path = root_path + '/docs/api/r/mxnet-r-reference-manual.pdf'
    _run_cmd('cd ' + r_root +
             '; R -e "roxygen2::roxygenize()"; R CMD Rd2pdf . --no-preview -o ' + pdf_path)
    dest_path = app.builder.outdir + '/api/r/'
    _run_cmd('mkdir -p ' + dest_path + '; mv ' + pdf_path + ' ' + dest_path)

def build_scala_docs(app):
    """build scala doc and then move the outdir"""
    scala_path = app.builder.srcdir + '/../scala-package/core/src/main/scala/ml/dmlc/mxnet'
    # scaldoc fails on some apis, so exit 0 to pass the check
    _run_cmd('cd ' + scala_path + '; scaladoc `find . | grep .*scala`; exit 0')
    dest_path = app.builder.outdir + '/api/scala/docs'
    _run_cmd('rm -rf ' + dest_path)
    _run_cmd('mkdir -p ' + dest_path)
    scaladocs = ['index', 'index.html', 'ml', 'lib', 'index.js', 'package.html']
    for doc_file in scaladocs:
        _run_cmd('cd ' + scala_path + ' && mv -f ' + doc_file + ' ' + dest_path)

def _convert_md_table_to_rst(table):
    """Convert a markdown table to rst format"""
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
                out += '   * - ' if j == 0 else '     - '
                out += pypandoc.convert_text(
                    c, 'rst', format='md').replace('\n', ' ').replace('\r', '') + '\n'
    out += '```\n'
    return out


def convert_table(app, docname, source):
    """Find tables in a markdown and then convert them into the rst format"""
    num_tables = 0
    for i,j in enumerate(source):
        table = []
        output = ''
        in_table = False
        for l in j.split('\n'):
            r = l.strip()
            if r.startswith('|'):
                table.append(r)
                in_table = True
            else:
                if in_table is True:
                    converted = _convert_md_table_to_rst(table)
                    if converted is '':
                        print("Failed to convert the markdown table")
                        print(table)
                    else:
                        num_tables += 1
                    output += converted
                    in_table = False
                    table = []
                output += l + '\n'
        source[i] = output
    if num_tables > 0:
        print('Converted %d tables in %s' % (num_tables, docname))

def _parse_code_lines(lines):
    """A iterator that returns if a line is within a code block

    Returns
    -------
    iterator of (str, bool, str, int)
        - line: the line
        - in_code: if this line is in a code block
        - lang: the code block langunage
        - indent: the code indent
    """
    in_code = False
    lang = None
    indent = None
    for l in lines:
        m = _CODE_MARK.match(l)
        if m is not None:
            if not in_code:
                if m.groups()[1].lower() in _LANGS:
                    lang = m.groups()[1].lower()
                    indent = len(m.groups()[0])
                    in_code = True
                yield (l, in_code, lang, indent)
            else:
                yield (l, in_code, lang, indent)
                lang = None
                indent = None
                in_code = False
        else:
            yield (l, in_code, lang, indent)

def _get_lang_selection_btn(langs):
    active = True
    btngroup = '<div class="text-center">\n<div class="btn-group opt-group" role="group">'
    for l in langs:
        btngroup += '<button type="button" class="btn btn-default opt %s">%s</button>\n' % (
            'active' if active else '', l[0].upper()+l[1:].lower())
        active = False
        btngroup += '</div>\n</div> <script type="text/javascript" src="../../_static/js/options.js"></script>'
    return btngroup

def _get_blocks(lang, lines):
    cur_block = []
    pre_in_code = None
    for (l, in_code, cur_lang, _) in _parse_code_lines(lines):
        if in_code and cur_lang != lang:
            in_code = False
        if in_code != pre_in_code:
            if pre_in_code and len(cur_block) >= 2:
                cur_block = cur_block[1:-1] # remove ```
            # remove empty lines at head
            while len(cur_block) > 0:
                if len(cur_block[0]) == 0:
                    cur_block.pop(0)
                else:
                    break
            # remove empty lines at tail
            while len(cur_block) > 0:
                if len(cur_block[-1]) == 0:
                    cur_block.pop()
                else:
                    break
            if len(cur_block):
                yield (pre_in_code, cur_block)
            cur_block = []
        cur_block.append(l)
        pre_in_code = in_code
    if len(cur_block):
        yield (pre_in_code, cur_block)

def _get_jupyter_notebook(lang, lines):
    cells = []
    for in_code, lines in _get_blocks(lang, lines):
        cell = {
            "cell_type": "code" if in_code else "markdown",
            "metadata": {},
            "source":  '\n'.join(lines)
        }
        if in_code:
             cell.update({
                 "outputs": [],
                 "execution_count": None,
             })
        cells.append(cell)
    ipynb = {"nbformat" : 4,
             "nbformat_minor" : 2,
             "metadata" : {"language":lang, "display_name":'', "name":''},
             "cells" : cells}
    return ipynb

def _get_source(lang, lines):
    cmt = _LANGS[lang][1] + ' '
    out = []
    for in_code, lines in _get_blocks(lang, lines):
        if in_code:
            out.append('')
        for l in lines:
            if in_code:
                out.append(l)
            else:
                if ('<div>' in l or '</div>' in l or
                    '<script>' in l or '</script>' in l or
                    '<!--' in l or '-->' in l):
                    continue
                out.append(cmt+l)
        if in_code:
            out.append('')

    return out

def _get_src_download_btn(out_prefix, langs, lines):
    btn = '<div class="btn-group" role="group">\n'
    for lang in langs:
        ipynb = out_prefix + '_' + lang + '.ipynb'
        with open(ipynb, 'w') as f:
            json.dump(_get_jupyter_notebook(lang, lines), f)
        src = out_prefix + '.' + _LANGS[lang][0]
        with open(src, 'w') as f:
            f.write('\n'.join(_get_source(lang, lines)))
        for f in [ipynb, src]:
            f = f.split('/')[-1]
            btn += '<button type="button" class="btn btn-default">'
            btn += '<a href="%s"><span class="glyphicon glyphicon-download-alt"></span> %s </a></button>\n' % (f, f)
    btn += '</div>\n'
    return btn

def add_buttons(app, docname, source):
    out_prefix = app.builder.outdir + '/' + docname
    dirname = os.path.dirname(out_prefix)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    for i,j in enumerate(source):
        lines = j.split('\n')
        langs = set([l for (_, _, l, _) in _parse_code_lines(lines)
                     if l is not None and l in _LANGS])
        # first convert
        for k,l in enumerate(lines):
            if _SRC_DOWNLOAD_MARK in l:
                lines[k] = _get_src_download_btn(
                    out_prefix, langs, lines)
        # then add lang buttons
        for k,l in enumerate(lines):
            if _LANG_SELECTION_MARK in l:
                lines[k] = _get_lang_selection_btn(langs)
        source[i] = '\n'.join(lines)

def setup(app):
    app.connect("builder-inited", build_mxnet)
    app.connect("builder-inited", generate_doxygen)
    app.connect("builder-inited", build_scala_docs)
    # skipped to build r, it requires to install latex, which is kinds of too heavy
    # app.connect("builder-inited", build_r_docs)
    app.connect('source-read', convert_table)
    app.connect('source-read', add_buttons)
    app.add_config_value('recommonmark_config', {
        'url_resolver': lambda url: 'http://mxnet.io/' + url,
        'enable_eval_rst': True,
    }, True)
    app.add_transform(transform.AutoStructify)
