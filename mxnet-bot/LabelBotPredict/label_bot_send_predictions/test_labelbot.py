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
from LabelBot import LabelBot
# some version issue
try:
    from unittest.mock import patch
except ImportError:
    from mock import patch


class TestLabelBot(unittest.TestCase):
	"""
	Unittests for LabelBot.py
	Test coverage: 92%
	"""

	def setUp(self):
		self.lb = LabelBot(img_file="./test_img.png")
		self.lb.repo = "apache/incubator-mxnet"
		self.lb.sender = "fake@email.com"
		self.lb.recipients = ["fake2@email.com"]
		self.lb.aws_region = "us-east-1"
		self.lb.elastic_beanstalk_url = "http://fakedocker.us-west-2.elasticbeanstalk.com"


	def test_read_repo(self):
		with patch('LabelBot.requests.get') as mocked_get:
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
							  					"html_url":"https://github.com/apache/incubator-mxnet/issues/11924",
							  				  }]
			self.lb.read_repo(False)


	def test_sendemail(self):
		with patch('LabelBot.requests.get') as mocked_get, patch('LabelBot.requests.post') as mocked_post:
			mocked_get.return_value.status_code = 200
			mocked_get.return_value.json.return_value = [
											   {"body":"issue's body",
											 	"created_at":"2018-08-04T18:27:17Z",
							  					"comments":0,
							  					"number":11925,
							 					"labels":[{'name':'Doc'}],
							  					"state":"open",
							  					"title":"issue's title",
							  					"html_url":"https://github.com/apache/incubator-mxnet/issues/11925",
							  				  },
							  				  {"body":"issue's body",
											 	"created_at":"2018-08-04T18:27:17Z",
							  					"comments":1,
							  					"comments_url":"https://api.github.com/repos/apache/incubator-mxnet/issues/11918/comments",
							  					"number":11918,
							 					"labels":[],
							  					"state":"open",
							  					"title":"issue's title",
							  					"html_url":"https://github.com/apache/incubator-mxnet/issues/11918",
							  				  }]
			mocked_post.return_value.json.return_value = [{'number': 11919, 'predictions': ['Performance']}, 
														  {'number': 11924, 'predictions': ['Build']}]
			self.assertRaises(ClientError, self.lb.sendemail())


if __name__ == "__main__":
	unittest.main()