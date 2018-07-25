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

#pylint: disable=no-member, too-many-instance-attributes
"""This script uses pyenchant to check spelling for MXNet
    documentation website.
    An exclude list is provided to avoid checking specific word,
    such as NDArray.
"""
from __future__ import print_function

import os
import sys
import re
from HTMLParser import HTMLParser
import enchant
from enchant.checker import SpellChecker
import grammar_check
import html2text

try:
    reload(sys)  # Python 2
    sys.setdefaultencoding('utf-8')
except NameError:
    pass         # Python 3


GRAMMAR_CHECK_IGNORE = ['WHITESPACE_RULE', 'DOUBLE_PUNCTUATION', 'EN_QUOTES[1]',
                        'EN_QUOTES[2]', 'COMMA_PARENTHESIS_WHITESPACE',
                        'ENGLISH_WORD_REPEAT_RULE', 'EN_UNPAIRED_BRACKETS',
                        'ENGLISH_WORD_REPEAT_BEGINNING_RULE', 'CD_NN[1]',
                        'UPPERCASE_SENTENCE_START', 'ALL_OF_THE[1]', 'EN_QUOTES[3]',
                        'THREE_NN[1]', 'HE_VERB_AGR[7]', 'NUMEROUS_DIFFERENT[1]',
                        'LIFE_TIME[1]', 'PERIOD_OF_TIME[1]', 'WITH_OUT[1]', 'LARGE_NUMBER_OF[1]',
                        'MANY_NN_U[3]', 'COMP_THAN[3]', 'MASS_AGREEMENT[1]', 'MANY_NN[1]',
                        'GENERAL_XX[1]', 'EN_A_VS_AN']



def get_grammar_res(matches):
    """Filter the grammar check result with ignored check types.

       Parameters
       -----------
       matches: list
       Match result of grammar check

       Return
       ---------
       ret: list
       Filtered result
    """
    ret = []
    for match in matches:
        lines = str(match).split('\n')
        lines[0] = lines[0].rstrip()
        is_ignored = False
        for entry in GRAMMAR_CHECK_IGNORE:
            if lines[0].endswith(entry):
                is_ignored = True
                break
        if not is_ignored:
            ret.append(match)
    return ret



def check_doc(file_content, spell_checker, spell_check_ret):
    """A documentation checker checks spelling
       of files.

       Parameters
       -----------
       content: str
       source text to be checked

       spell_checker: enchant.checker.SpellChecker
       Spell checker

       spell_check_res: dict
       Spell check result dictionary maps typo word to occurance times.
    """
    spell_checker.set_text(file_content)
    for error in spell_checker:
        if error.word in spell_check_ret:
            spell_check_ret[error.word] += 1
        else:
            spell_check_ret[error.word] = 1


class DocParser(HTMLParser):
    """A document parser parsed html file and conduct spelling check
        and grammar check.
    """
    def __init__(self):
        HTMLParser.__init__(self)
        self.__spell_check_res = {}
        self.__grammar_check_res = None
        self.__ignore_tag = False
        self.__is_code_block = False
        self.__in_code_block = False
        self.__dictionary = enchant.DictWithPWL('en_US', 'web-data/mxnet/doc/ignored_words.txt')
        self.__spell_checker = SpellChecker(self.__dictionary)
        self.__parsed_content = ""
        self.__grammar_checker = grammar_check.LanguageTool('en-US')

    def handle_starttag(self, tag, attrs):
        self.__ignore_tag = True if tag.startswith('script') or tag.startswith('option') else False

    def handle_endtag(self, tag):
        pass

    def handle_data(self, data):
        #Ignore url content
        if not self.__ignore_tag and not data.startswith('http'):
            check_doc(data, self.__spell_checker, self.__spell_check_res)


    def get_res(self):
        """return the checking result
        """
        return [self.__spell_check_res, self.__grammar_check_res]


    def clear_res(self):
        """Clean the checking result
        """
        self.__spell_check_res = {}
        self.__grammar_check_res = None


    def check_grammar(self, file_name):
        """Check the grammar of the specified file

           Parameters
           -----------
           file_name: name of the file to be checked
        """
        file_content = html2text.html2text(open(file_name).read())
        file_content = re.sub(u"[\x00-\x08\x0b-\x0c\x0e-\x1f]+", u"", file_content)
        self.__grammar_check_res = self.__grammar_checker.check(file_content)


if __name__ == "__main__":
    BUILD_HTML_DIR = '../../../docs/_build/html'
    CHINESE_HTML_DIR = '../../../docs/_build/html/zh'
    STATIC_HTML_DIR = '../../../docs/_build/html/_static'
    DOC_PARSER = DocParser()
    ALL_CLEAR = True
    for root, _, files in os.walk(BUILD_HTML_DIR):
        if root.startswith(CHINESE_HTML_DIR) or root.startswith(STATIC_HTML_DIR):
            continue
        for read_file in files:
            if not read_file.endswith('.html') or read_file == 'README.html' or '_zh' in read_file:
                continue
            rd_file = open(os.path.join(root, read_file), 'r')
            content = rd_file.read()
            DOC_PARSER.clear_res()
            DOC_PARSER.feed(content)
            DOC_PARSER.check_grammar(os.path.join(root, read_file))
            spell_check_res = DOC_PARSER.get_res()[0]
            grammar_check_res = DOC_PARSER.get_res()[1]
            if len(spell_check_res) > 0:
                print("%s has typo:" % os.path.join(root, read_file))
                print("%s\n" % spell_check_res)
                ALL_CLEAR = False
    if ALL_CLEAR:
        print("No typo is found.")
