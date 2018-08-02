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

import requests
import unittest
import boto3
import pandas as pd
from botocore.exceptions import ClientError
from botocore.exceptions import NoCredentialsError
from SentenceParser import SentenceParser
from pandas.util.testing import assert_frame_equal
# some version issue
try:
    from unittest.mock import patch
except ImportError:
    from mock import patch

# test coverage: 88%
class TestSentenceParser(unittest.TestCase):

	def setUp(self):
		self.sp = SentenceParser()
		self.sp.data = pd.DataFrame([{'id': 11925, 'title': "issue's title", 
									  'body': " bug ``` import pandas``` ## Environment info", 
									  'labels': ['Doc']}])

	def tearDown(self):
		pass

	def test_read_file(self):
		self.sp.read_file('all_data.json_all_labels', 'json')
		expected_data = [{'id': 11925, 'title': "issue's title", 'body': "issue's body", 'labels': ['Doc']}, 
						 {'id': 11924, 'title': "issue's title", 'body': "issue's body", 'labels': []}]
		assert_frame_equal(self.sp.data, pd.DataFrame(expected_data))

	def test_merge_column(self):						 
		self.sp.merge_column(['title', 'body'], 'train')
		expected_data = [{'id': 11925, 'title': "issue's title", 'body': " bug ``` import pandas``` ## Environment info", 
						  'labels': ['Doc'],
						  'train': " issue's title  bug ``` import pandas``` ## Environment info"}]
		assert_frame_equal(self.sp.data, pd.DataFrame(expected_data))

	def test_clean_body(self):
		self.sp.clean_body('body', True, True)
		expected_data = [{'id': 11925, 'title': "issue's title", 'body': " bug   ", 'labels': ['Doc']}]
		assert_frame_equal(self.sp.data, pd.DataFrame(expected_data))

	def test_process_text(self):
		data = self.sp.process_text('body', True, True, True)
		expected_data = ['bug import panda environ info']
		self.assertEqual(data, expected_data)


if __name__ == "__main__":
	unittest.main()