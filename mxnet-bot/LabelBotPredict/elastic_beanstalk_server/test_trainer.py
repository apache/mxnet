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
from DataFetcher import DataFetcher
from Trainer import Trainer
# some version issue
try:
    from unittest.mock import patch
except ImportError:
    from mock import patch

# test coverage: 100%
class TestTrainer(unittest.TestCase):

	def setUp(self):
		self.trainer = Trainer()

	def test_train(self):
		with patch('DataFetcher.requests.get') as mocked_get:
			mocked_get.return_value.status_code = 200
			mocked_get.return_value.json.return_value = [{ "body":"I was looking at the mxnet.\
												metric source code and documentation",
											 	"created_at":"2018-07-28T18:27:17Z",
							  					"comments":"0",
							  					"number":11925,
							 					"labels":[{'name':'Doc'}],
							  					"state":"open",
							  					"title":"Confusion in documentation/implementation of F1, MCC metrics",
							  					"html_url":"https://github.com/apache/incubator-mxnet/issues/11925",
							  				  },
							  				  { "body":"I train a CNN with python under mxnet gluon mys C++ code crash when i call MXPredsetInput.",
											 	"created_at":"2018-07-28T18:27:17Z",
							  					"comments":"0",
							  					"number":11924,
							 					"labels":[{'name':'Bug'}],
							  					"state":"closed",
							  					"title":"Issue in exporting gluon model",
							  					"html_url":"https://github.com/apache/incubator-mxnet/issues/11924",
							  				  }]
			self.trainer.train()

if __name__ == "__main__":
	unittest.main()