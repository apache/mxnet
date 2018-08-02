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
from LabelBot import LabelBot
# some version issue
try:
    from unittest.mock import patch
except ImportError:
    from mock import patch

# test coverage: 92%
class TestLabelBot(unittest.TestCase):

	def setUp(self):
		self.lb = LabelBot("./test_img.png")
		self.lb.REPO = "apache/incubator-mxnet"
		self.lb.sender = "fake@email.com"
		self.lb.recipients = ["fake2@email.com"]
		self.lb.elastic_beanstalk_url = "http://fakedocker.us-west-2.elasticbeanstalk.com"

	def tearDown(self):
		pass

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
							  					"html_url":"https://github.com/apache/incubator-mxnet/issues/11925",
							  				  }]
			self.lb.read_repo(True)

	def test_sendemail(self):
		with patch('LabelBot.requests.get') as mocked_get, patch('LabelBot.requests.post') as mocked_post:
			mocked_get.return_value.status_code = 200
			mocked_get.return_value.json.return_value = [
											   {"body":"issue's body",
											 	"created_at":"2018-07-28T18:27:17Z",
							  					"comments":0,
							  					"number":11925,
							 					"labels":[{'name':'Doc'}],
							  					"state":"open",
							  					"title":"issue's title",
							  					"html_url":"https://github.com/apache/incubator-mxnet/issues/11925",
							  				  },
							  				  {"body":"issue's body",
											 	"created_at":"2018-07-28T18:27:17Z",
							  					"comments":1,
							  					"comments_url":"https://api.github.com/repos/apache/incubator-mxnet/issues/11918/comments",
							  					"number":11918,
							 					"labels":[],
							  					"state":"open",
							  					"title":"issue's title",
							  					"html_url":"https://github.com/apache/incubator-mxnet/issues/11918",
							  				  }]
			#mocked_post.return_value.status_code = 200
			mocked_post.return_value.json.return_value = [{'number': 11919, 'predictions': ['Performance']}, 
														  {'number': 11924, 'predictions': ['Build']}]
			self.assertRaises(NoCredentialsError, self.lb.sendemail())


if __name__ == "__main__":
	unittest.main()