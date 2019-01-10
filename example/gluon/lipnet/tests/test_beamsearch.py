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

"""it is the test for the decode using beam search
Ref:
https://github.com/ThomasDelteil/HandwrittenTextRecognition_MXNet/blob/master/utils/CTCDecoder/BeamSearch.py
"""

import unittest
import numpy as np
from BeamSearch import ctcBeamSearch

class TestBeamSearch(unittest.TestCase):
    """Test Beam Search
    """
    def test_ctc_beam_search(self):
        "test decoder"
        classes = 'ab'
        mat = np.array([[0.4, 0, 0.6], [0.4, 0, 0.6]])
        print('Test beam search')
        expected = 'a'
        actual = ctcBeamSearch(mat, classes, None, k=2, beamWidth=3)[0]
        print('Expected: "' + expected + '"')
        print('Actual: "' + actual + '"')
        self.assertEqual(expected, actual)

if __name__ == '__main__':
    unittest.main()
