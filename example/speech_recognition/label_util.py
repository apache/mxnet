# -*- coding: utf-8 -*-

import csv

from log_util import LogUtil
from singleton import Singleton


@Singleton
class LabelUtil:
    _log = None

    # dataPath
    def __init__(self):
        self._log = LogUtil().getlogger()
        self._log.debug("LabelUtil init")

    def load_unicode_set(self, unicodeFilePath):
        self.byChar = {}
        self.byIndex = {}
        self.unicodeFilePath = unicodeFilePath

        with open(unicodeFilePath) as data_file:
            data_file = csv.reader(data_file, delimiter=',')

            self.count = 0
            for r in data_file:
                self.byChar[r[0]] = int(r[1])
                self.byIndex[int(r[1])] = r[0]
                self.count += 1


    def to_unicode(self, src, index):
        # 1 byte
        code1 = int(ord(src[index + 0]))

        index += 1

        result = code1

        return result, index

    def convert_word_to_grapheme(self, label):

        result = []

        index = 0
        while index < len(label):
            (code, nextIndex) = self.to_unicode(label, index)

            result.append(label[index])

            index = nextIndex

        return result, "".join(result)

    def convert_word_to_num(self, word):
        try:
            label_list, _ = self.convert_word_to_grapheme(word)

            label_num = []

            for char in label_list:
                # skip word
                if char == "":
                    pass
                else:
                    label_num.append(int(self.byChar[char]))

            # tuple typecast: read only, faster
            return tuple(label_num)

        except AttributeError:
            self._log.error("unicodeSet is not loaded")
            exit(-1)

        except KeyError as err:
            self._log.error("unicodeSet Key not found: %s" % err)
            exit(-1)

    def convert_bi_graphemes_to_num(self, word):
            label_num = []

            for char in word:
                # skip word
                if char == "":
                    pass
                else:
                    label_num.append(int(self.byChar[char]))

            # tuple typecast: read only, faster
            return tuple(label_num)


    def convert_num_to_word(self, num_list):
        try:
            label_list = []
            for num in num_list:
                label_list.append(self.byIndex[num])

            return ''.join(label_list)

        except AttributeError:
            self._log.error("unicodeSet is not loaded")
            exit(-1)

        except KeyError as err:
            self._log.error("unicodeSet Key not found: %s" % err)
            exit(-1)

    def get_count(self):
        try:
            return self.count

        except AttributeError:
            self._log.error("unicodeSet is not loaded")
            exit(-1)

    def get_unicode_file_path(self):
        try:
            return self.unicodeFilePath

        except AttributeError:
            self._log.error("unicodeSet is not loaded")
            exit(-1)

    def get_blank_index(self):
        return self.byChar["-"]

    def get_space_index(self):
        return self.byChar["$"]
