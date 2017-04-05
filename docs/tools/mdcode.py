#!/usr/bin/env python
"""Manipulate codes blocks in a markdown file"""
import argparse
import re
import os
import codecs
import json

# language names and the according file extensions
_LANGS = {'python':'py', 'r':'R', 'scala':'scala', 'julia':'jl', 'perl':'pl', 'cpp':'cc'}

# start or end of a code block
_CODE_MARK = re.compile('^([ ]*)```([\w]*)')

class Markdown2Notebook(object):
    """
    Convert markdown file to jupyter notebook. Support multiple
    languages and kernels.
    """

    #Notebook cell types
    markdown = 'markdown'
    code = 'code'

    execution_count = 0

    def parse_markdown(self, input, output, language):
        """Convert a markdown file to a jupyter notebook
           with corresponding language and kernel.

        Parameters
        -----------
        input : string
        Markdown file to be converted

        output : string
        Output notebook file

        language : string
        language of notebook
        """
        with open(input, 'r') as md_file:

            notebook = open(output, 'w')
            text = md_file.read()
            text = text.replace("```bash", "```sh")
            code_flag = re.compile('```')
            nb_content = {'cells':[],
                          'metadata': {
                              'kernelspec': {
                               'language': language,
                                'name': '',
                                'display_name': ''
                           }
                          },
                          'nbformat': 4,
                          'nbformat_minor': 2
                         }
            markdown_start = 0
            code_start = 0
            is_lang_block = False

            for index, match in enumerate(code_flag.finditer(text)):
                if index % 2 != 0:
                    if is_lang_block:
                        markdown_start = match.span()[1] + 1
                        code_end = match.span()[0] - 1
                        #Parse current code block and write to notebook
                        if code_start >= 0 and code_start < code_end:
                            self.execution_count += 1
                            self._parse_block(nb_content,
                                              text[code_start:code_end], self.code)

                else:
                    for key, _ in _LANGS.items():
                        if text[match.span()[0]:].startswith("```" + key):
                            is_lang_block = True
                            break
                        else:
                            is_lang_block = False
                    markdown_end = match.span()[0] - 1
                    if is_lang_block:
                        self._parse_block(nb_content,
                                          text[markdown_start:markdown_end], self.markdown)
                    if text[match.span()[0]:].startswith("```" + language):
                        code_start = match.span()[0] + len(language) + 4
                    else:
                        code_start = -1

            json.dump(nb_content, notebook)
            notebook.close()

    def _parse_block(self, content, text, cell_type):
        """Parse a block of markdown or code into a notebook cell

        Parameters
        -----------
        content : dict
        Dictionary contains notebook content

        text : string
        text of markdown or code to be parsed

        cell_type : string
        cell type of 'markdown' or 'code'
        """
        cell = {'cell_type': cell_type,
                "metadata": {},
                'source': []}
        lines = text.splitlines()
        if cell_type == 'code':
            cell['outputs'] = []
            cell['execution_count'] = self.execution_count
            leading_space = 0
            for line in lines:
                if len(line) > 0:
                    leading_space = len(line) - len(line.lstrip())
                    break
            for line in lines:
                cell['source'].append(line[leading_space:] + '\n')
        else:
            for line in lines:
                cell['source'].append(line + '\n')
        # Remove last line break
        cell['source'][-1] = cell['source'][-1][:-1]
        content['cells'].append(cell)

class CodeBlocks(object):
    def __init__(self, fname, lang):
        with codecs.open(fname, 'r', 'utf-8') as f:
            self.data = f.readlines()
        self.lang = lang.lower()
        self.cells = []
        self.converter = Markdown2Notebook()
        self.input_file = fname

    def _parse_lines(self):
        in_code = False
        lang = None
        indent = None
        for l in self.data:
            m = _CODE_MARK.match(l)
            if m is not None:
                if not in_code:
                    lang = m.groups()[1].lower()
                    indent = len(m.groups()[0])
                yield (l, True, lang, indent)
                if in_code:
                    lang = None
                    indent = None
                in_code = not in_code
            else:
                yield (l, in_code, lang, indent)

    def _add_jupyter_block(self, lines, is_code ):
        if is_code and len(lines) >= 2:
            lines = lines[1:-1] # remove ```
        while len(lines) > 0:
            if len(lines[0].rstrip()) == 0:
                lines.pop(0)
            else:
                break
        while len(lines) > 0:
            if len(lines[-1].rstrip()) == 0:
                lines.pop()
            else:
                break
        if len(lines) == 0:
            return
        lines[-1] = lines[-1].rstrip()
        cell = {
            "cell_type": "code" if is_code else "markdown",
            "metadata": {},
            "source":  lines
        }
        if is_code:
            cell.update({
                "outputs": [],
                "execution_count": None,
            })
        self.cells.append(cell)

    def write(self, action, ofname):
        if action == 'get':
            with open(ofname, 'w') as f:
                for (l, in_code, lang, indent) in self._parse_lines():
                    if in_code and lang == self.lang and l[indent:indent+3] != '```':
                        f.write(l[indent:])
            return
        if action == 'keep':
            with open(ofname, 'w') as f:
                for (l, in_code, lang, _) in self._parse_lines():
                    if not in_code or in_code and lang == self.lang:
                        f.write(l)
            return
        if action == 'convert':
            self.converter.parse_markdown(self.input_file, ofname, self.lang)
            return
        if action == 'add_btn':
            langs = set([l for (_, _, l, _) in self._parse_lines() if l is not None])
            print langs
            active = True
            btngroup = """<div class="text-center">
<div class="btn-group opt-group" role="group">
"""
            for l in langs:
                btngroup += "<button type=\"button\" class=\"btn btn-default opt %s\">%s</button>\n" % (
                    'active' if active else '', l[0].upper()+l[1:].lower())
                active = False
            btngroup += """</div>
</div>
<script type="text/javascript" src='../../_static/js/options.js'></script>
"""
            with open(ofname, 'w') as f:
                for l in self.data:
                    if 'ENABLE LANGUAGE BAR' in l:
                        f.write(btngroup)
                    else:
                        f.write(l)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""Manipulate code blocks in markdown files.

Sample usage:

- extract all python code blocks in example

  ./mdcode.py get python example.md example.py

- remove all codes blocks except for python

  ./mdcode.py keep python example.md example_py.md

- remove all codes blocks except for python and then convert into jupyter notebook

  ./mdcode.py convert python example.md example.ipynb

- add the language selection botton group into example.md

  ./mdcode.py add_btn all example.md example.md
    """)
    parser.add_argument('action', help='action',
                        choices=['get', 'keep', 'convert', 'add_btn'])
    parser.add_argument('lang', help='code language')
    parser.add_argument('input', help='input markdown filename')
    parser.add_argument('output', help='output file')
    args = parser.parse_args()

    code_blocks = CodeBlocks(args.input, args.lang)
    code_blocks.write(args.action, args.output)
