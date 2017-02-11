#pylint: disable=
"""
    This script uses pyenchant to check spelling for MXNet
    documentation website.
    An exclude list is provided to avoid checking specific word,
    such as NDArray.
"""

import os
import re
from HTMLParser import HTMLParser
import enchant
from enchant.checker import SpellChecker
import grammar_check
import html2text

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

word_list = set()
GRAMMAR_CHECK_IGNORE = ['WHITESPACE_RULE', 'DOUBLE_PUNCTUATION', 'EN_QUOTES[1]',
                                                      'EN_QUOTES[2]', 'COMMA_PARENTHESIS_WHITESPACE',
                                                      'ENGLISH_WORD_REPEAT_RULE', 'EN_UNPAIRED_BRACKETS',
                                                      'ENGLISH_WORD_REPEAT_BEGINNING_RULE', 'CD_NN[1]',
                                                      'UPPERCASE_SENTENCE_START', 'ALL_OF_THE[1]', 'EN_QUOTES[3]',
                                                      'THREE_NN[1]', 'HE_VERB_AGR[7]', 'NUMEROUS_DIFFERENT[1]',
                                                      'LIFE_TIME[1]', 'PERIOD_OF_TIME[1]', 'WITH_OUT[1]', 'LARGE_NUMBER_OF[1]',
                                                      'MANY_NN_U[3]', 'COMP_THAN[3]', 'MASS_AGREEMENT[1]', 'MANY_NN[1]',
                                                      'GENERAL_XX[1]', 'EN_A_VS_AN']


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
def get_grammar_res(matches):
    ret = []
    for match in matches:
        lines = str(match).split('\n')
        lines[0] = lines[0].rstrip()
        is_ignored  = False
        for item in GRAMMAR_CHECK_IGNORE:
            if lines[0].endswith(item):
                is_ignored = True
                break
        if not is_ignored:
            ret.append(match)
    return ret

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
def check_doc(content, spell_checker, spell_check_res):
    spell_checker.set_text(content)
    for error in spell_checker:
        if spell_check_res.has_key(error.word):
            spell_check_res[error.word] += 1
        else:
            spell_check_res[error.word] = 1
        word_list.add(error.word)


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
        self.__dictionary = enchant.DictWithPWL('en_US', 'ignored_words.txt')
        self.__spell_checker = SpellChecker(self.__dictionary)
        self.__parsed_content = ""
        self.__grammar_checker = grammar_check.LanguageTool('en-US')
        
    def handle_starttag(self, tag, attrs):
        self.__ignore_tag = True if tag.startswith('script') or tag.startswith('option') else False
        """If is_code_block flag is true, we're in a code block.
            in_code_block is set.
            if is_code_block is false, we check the class attribute
            to decide whether this is a start tag for code block.
        """
        if self.__is_code_block:
            self.__in_code_block = True
            return
        for attr in attrs:
            if attr[0] == 'class' and attr[1] in CODE_BLOCK_CLASS:
                self.__is_code_block = True
                break
            
    def handle_endtag(self, tag):
        if self.__is_code_block and not self.__in_code_block:
            self.__is_code_block = False
        if self.__is_code_block and self.__in_code_block:
            self.__in_code_block = False
            
    def handle_data(self, data):
        #Ignore url content
        if not self.__ignore_tag and not data.startswith('http'):
            check_doc(data, self.__spell_checker, self.__spell_check_res)
            if self.__is_code_block:
                return
            line_list = data.split('\n')
            for line in line_list:
                line = line.lstrip()
                line = line.rstrip()
                if len(line) == 0:
                    continue
                if line[0] == ',' or line[0] == '.' or line[0] == ':' or line[0] == ';'  :
                    self.__parsed_content = self.__parsed_content[:-1]
                self.__parsed_content += line + ' '

    def __get_parsed_content(self):
        return self.__parsed_content
    
    def clear_parsed_content(self):
        self.__parsed_content = ""
    
    def get_res(self):
        return [self.__spell_check_res, self.__grammar_check_res]

    def clear_res(self):
        self.__spell_check_res = {}
        self.__grammar_check_res = None

    def check_grammar(self, file_name):
        content = html2text.html2text(open(file_name).read())
        content = re.sub(u"[\x00-\x08\x0b-\x0c\x0e-\x1f]+",u"", content)
        self.__grammar_check_res = self.__grammar_checker.check(content)
        

if __name__ == "__main__":
    build_html_dir = '../../../docs/_build/html'
    chinese_html_dir = '../../../docs/_build/html/zh'
    static_html_dir = '../../../docs/_build/html/_static'
    doc_parser = DocParser()
    res = open('result.txt', 'w')
    for root, _, files in os.walk(build_html_dir):
        if root.startswith(chinese_html_dir) or root.startswith(static_html_dir):
            continue
        for read_file in files:
            if not read_file.endswith('.html') or read_file == 'README.html' or '_zh' in read_file:
                continue
            rd_file = open(os.path.join(root, read_file), 'r')
            content = rd_file.read()
            doc_parser.feed(content)
            doc_parser.check_grammar(os.path.join(root, read_file))
            spell_check_res = doc_parser.get_res()[0]
            grammar_check_res = doc_parser.get_res()[1]
            if len(spell_check_res) > 0:
                print "%s has typo:" % os.path.join(root, read_file)
                print "%s\n" % spell_check_res
            if grammar_check_res:
                filtered_res = get_grammar_res(grammar_check_res)
                if len(filtered_res) > 0:
                    print "%s has grammar issue:" % os.path.join(root, read_file)
                    res.write("%s has grammar issue:\n" % os.path.join(root, read_file))
                    for item in filtered_res:
                        print "%s\n" % item
                        res.write("%s\n" % item)
            doc_parser.clear_res()
            doc_parser.clear_parsed_content()
    res.close()
    for word in word_list:
        print word
