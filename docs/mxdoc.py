"""A sphnix-doc plugin to build mxnet docs"""
import subprocess
from recommonmark import transform
import pypandoc

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

def generate_doxygen_xml(app):
    """Run the doxygen make commands"""
    _run_cmd("cd %s/.. && make doxygen" % app.builder.srcdir)
    _run_cmd("cp -rf doxygen/html %s/doxygen" % app.builder.outdir)

def build_mxnet(app):
    """Build mxnet .so lib"""
    _run_cmd("cd %s/.. && cp make/config.mk config.mk && make -j$(nproc)" %
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

def setup(app):
    app.connect("builder-inited", build_mxnet)
    # skipped to build c api doc
    # app.connect("builder-inited", generate_doxygen_xml)
    # app.connect("builder-inited", build_scala_docs)
    # skipped to build r, it requires to install latex, which is kinds of too heavy
    # app.connect("builder-inited", build_r_docs)

    app.connect('source-read', convert_table)
    app.add_config_value('recommonmark_config', {
        'url_resolver': lambda url: 'https://github.com/dmlc/mxnet/tree/master/docs/' + url,
        'enable_eval_rst': True,
    }, True)
    app.add_transform(transform.AutoStructify)
