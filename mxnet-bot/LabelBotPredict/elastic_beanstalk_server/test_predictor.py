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
from botocore.exceptions import ClientError
from botocore.exceptions import NoCredentialsError
from Predictor import Predictor
from DataFetcher import DataFetcher
from SentenceParser import SentenceParser
# some version issue
try:
    from unittest.mock import patch
except ImportError:
    from mock import patch

# test coverage: 100%
class TestLabelBot(unittest.TestCase):

	def setUp(self):
		self.pr = Predictor()

	def tearDown(self):
		pass

	def test_tokenize(self):
		words = self.pr.tokenize("hello_world")
		self.assertEqual(words, set(['hello','world']))

	def test_rule_based(self):
		with patch('DataFetcher.requests.get') as mocked_get:
			mocked_get.return_value.status_code = 200
			mocked_get.return_value.json.return_value = { "body":"issue's body",
											 	"created_at":"2018-07-28T18:27:17Z",
							  					"comments":"0",
							  					"number":11925,
							 					"labels":[{'name':'Doc'}],
							  					"state":"open",
							  					"title":"a feature requests for scala package",
							  					"html_url":"https://github.com/apache/incubator-mxnet/issues/11925",
							  				  }
			predictions = self.pr.rule_based([11925])
			self.assertEqual([['Feature','scala']], predictions)

	def test_ml_predict(self):
		with patch('DataFetcher.requests.get') as mocked_get:
			mocked_get.return_value.status_code = 200
			mocked_get.return_value.json.return_value = { "body":"test",
											 	"created_at":"2018-07-28T18:27:17Z",
							  					"comments":"0",
							  					"number":11925,
							 					"labels":[{'name':'Doc'}],
							  					"state":"open",
							  					"title":"a feature requests for scala package",
							  					"html_url":"https://github.com/apache/incubator-mxnet/issues/11925",
							  				  }
			predictions=self.pr.ml_predict([11925])
			self.assertEqual([['Feature']], predictions)

	def test_predict(self):
		with patch('DataFetcher.requests.get') as mocked_get:
			mocked_get.return_value.status_code = 200
			mocked_get.return_value.json.return_value = { "body":"test",
											 	"created_at":"2018-07-28T18:27:17Z",
							  					"comments":"0",
							  					"number":11925,
							 					"labels":[{'name':'Doc'}],
							  					"state":"open",
							  					"title":"a feature requests for scala package",
							  					"html_url":"https://github.com/apache/incubator-mxnet/issues/11925",
							  				  }
			predictions = self.pr.predict([11925])
			self.assertEqual([['Feature','scala']], predictions)

if __name__ == "__main__":
	unittest.main()