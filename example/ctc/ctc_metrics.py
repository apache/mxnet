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
"""Contains a class for calculating CTC eval metrics"""

from __future__ import print_function

import numpy as np


class CtcMetrics(object):
    """Module for calculating the prediction accuracy during training. Two accuracy measures are implemented:
    A simple accuracy measure that calculates number of correct predictions divided by total number of predictions
    and a second accuracy measure based on sum of Longest Common Sequence(LCS) ratio of all predictions divided by total
    number of predictions
    """
    def __init__(self, seq_len):
        self.seq_len = seq_len

    @staticmethod
    def ctc_label(p):
        """Iterates through p, identifying non-zero and non-repeating values, and returns them in a list Parameters
        ----------
        p: list of int

        Returns
        -------
        list of int
        """
        ret = []
        p1 = [0] + p
        for i, _ in enumerate(p):
            c1 = p1[i]
            c2 = p1[i+1]
            if c2 in (0, c1):
                continue
            ret.append(c2)
        return ret

    @staticmethod
    def _remove_blank(l):
        """ Removes trailing zeros in the list of integers and returns a new list of integers"""
        ret = []
        for i, _ in enumerate(l):
            if l[i] == 0:
                break
            ret.append(l[i])
        return ret

    @staticmethod
    def _lcs(p, l):
        """ Calculates the Longest Common Subsequence between p and l (both list of int) and returns its length"""
        # Dynamic Programming Finding LCS
        if len(p) == 0:
            return 0
        P = np.array(list(p)).reshape((1, len(p)))
        L = np.array(list(l)).reshape((len(l), 1))
        M = np.ndarray(shape=(len(P), len(L)), dtype=np.int32)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                up = 0 if i == 0 else M[i-1, j]
                left = 0 if j == 0 else M[i, j-1]

                if i == 0 or j == 0:
                    M[i, j] = max(up, left, M[i, j])
                else:
                    M[i, j] = M[i, j] + M[i - 1, j - 1]
        return M.max()

    def accuracy(self, label, pred):
        """ Simple accuracy measure: number of 100% accurate predictions divided by total number """
        hit = 0.
        total = 0.
        batch_size = label.shape[0]
        for i in range(batch_size):
            l = self._remove_blank(label[i])
            p = []
            for k in range(self.seq_len):
                p.append(np.argmax(pred[k * batch_size + i]))
            p = self.ctc_label(p)
            if len(p) == len(l):
                match = True
                for k, _ in enumerate(p):
                    if p[k] != int(l[k]):
                        match = False
                        break
                if match:
                    hit += 1.0
            total += 1.0
        assert total == batch_size
        return hit / total

    def accuracy_lcs(self, label, pred):
        """ Longest Common Subsequence accuracy measure: calculate accuracy of each prediction as LCS/length"""
        hit = 0.
        total = 0.
        batch_size = label.shape[0]
        for i in range(batch_size):
            l = self._remove_blank(label[i])
            p = []
            for k in range(self.seq_len):
                p.append(np.argmax(pred[k * batch_size + i]))
            p = self.ctc_label(p)
            hit += self._lcs(p, l) * 1.0 / len(l)
            total += 1.0
        assert total == batch_size
        return hit / total
