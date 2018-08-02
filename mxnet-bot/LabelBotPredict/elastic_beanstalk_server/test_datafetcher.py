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
from DataFetcher import DataFetcher
from pandas.util.testing import assert_frame_equal
# some version issue
try:
    from unittest.mock import patch
except ImportError:
    from mock import patch

# test coverage: 93%
class TestLabelBot(unittest.TestCase):

	def setUp(self):
		self.df = DataFetcher()
		self.df.REPO = "apache/incubator-mxnet"
		self.df.GITHUB_USER = "cathy"
		self.df.GITHUB_OAUTH_TOKEN = "123"

	def tearDown(self):
		pass

	def test_cleanstr(self):
		new_string = self.df.cleanstr("a_b", "")
		self.assertEqual(new_string, "ab")

	def test_count_pages(self):
		with patch('DataFetcher.requests.get') as mocked_get:
			mocked_get.return_value.status_code = 200
			mocked_get.return_value.json.return_value = [{ "body":"issue's body",
											 	"created_at":"2018-07-28T18:27:17Z",
							  					"comments":"0",
							  					"number":11925,
							 					"labels":[{'name':'Doc'}],
							  					"state":"open",
							  					"title":"issue's title",
							  					"html_url":"https://github.com/apache/incubator-mxnet/issues/11925",
							  				  },
							  				  { "body":"issue's body",
											 	"created_at":"2018-07-28T18:27:17Z",
							  					"comments":"0",
							  					"number":11924,
							 					"labels":[],
							  					"state":"closed",
							  					"title":"issue's title",
							  					"html_url":"https://github.com/apache/incubator-mxnet/issues/11925",
							  				  }]
			page = self.df.count_pages('all')
			self.assertEqual(page,1)

	def test_fetch_issues(self):
		with patch('DataFetcher.requests.get') as mocked_get:
			mocked_get.return_value.status_code = 200
			mocked_get.return_value.json.return_value = { "body":"issue's body",
											 	"created_at":"2018-07-28T18:27:17Z",
							  					"comments":"0",
							  					"number":11925,
							 					"labels":[{'name':'Feature'}],
							  					"state":"open",
							  					"title":"issue's title",
							  					"html_url":"https://github.com/apache/incubator-mxnet/issues/11925",
							  				  }
			data = self.df.fetch_issues([11925])
			expected_data = [{'id':"11925", 'title':"issue's title",'body':"issue's body"}]
			assert_frame_equal(data, pd.DataFrame(expected_data))

	def test_data2json(self):
		with patch('DataFetcher.requests.get') as mocked_get:
			mocked_get.return_value.status_code = 200
			mocked_get.return_value.json.return_value = [{ "body":"issue's body",
											 	"created_at":"2018-07-28T18:27:17Z",
							  					"comments":"0",
							  					"number":11925,
							 					"labels":[{'name':'Feature'}],
							  					"state":"open",
							  					"title":"issue's title",
							  					"html_url":"https://github.com/apache/incubator-mxnet/issues/11925",
							  				  },
							  				  { "body":"issue's body",
											 	"created_at":"2018-07-28T18:27:17Z",
							  					"comments":"0",
							  					"number":11924,
							 					"labels":[],
							  					"state":"closed",
							  					"title":"issue's title",
							  					"html_url":"https://github.com/apache/incubator-mxnet/issues/11925",
							  				  }]
			self.df.data2json('all', labels=["Feature"], other_labels=False)
			expected_data = [{'id': 11925, 'title': "issue's title", 'body': "issue's body", 'labels': 'Feature'}] 						 
			self.assertEqual(expected_data, self.df.json_data)

if __name__ == "__main__":
	unittest.main()