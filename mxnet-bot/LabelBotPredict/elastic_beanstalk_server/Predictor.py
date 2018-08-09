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

from sklearn.preprocessing           import LabelEncoder
from SentenceParser                  import SentenceParser
from DataFetcher                     import DataFetcher
import numpy  as np
import pickle
import re
import logging

logging.basicConfig(level=logging.INFO)


class Predictor:
	# keywords will be used to apply rule-based algorithms
	keywords = {"ci":["ci", "ccache", "jenkins"],
				"flaky":["flaky"],
				"gluon":["gluon"],
				"coda":["cuda", "cudnn"],
				"scala":["scala"],
				"mkldnn":["mkldnn, mkl"],
				"onnx":["onnx"]}


	def __init__(self):
		"""
		Predictor serves to apply rule-based and ML algorithms to predict labels
		"""
		self.tv = None
		self.labels = None
		self.clf = None
		self.reload()


	def reload(self):
		"""
		This method is to load models
		"""
		self.tv = pickle.load(open("/tmp/Vectorizer.p", "rb"))
		self.labels = pickle.load(open("/tmp/Labels.p", "rb"))
		self.clf = pickle.load(open("/tmp/Classifier.p", "rb"))


	def tokenize(self, row):
		"""
		This method is to tokenize a sentence into a list of words
		Args:
			row(string): a sentence
		Return:
			words(list): a list of words
		"""
		row = re.sub('[^a-zA-Z0-9]', ' ', row).lower()
		words = set(row.split())
		return words


	def rule_based(self, issues):
		"""
		This method applies rule_based algorithms to predict labels
		Args: 
			issues(list): a list of issue numbers
		Return:
		 	rule_based_predictions(list of lists): labels which satisfy rules
		"""
		DF = DataFetcher()
		df_test = DF.fetch_issues(issues)
		rule_based_predictions = []
		for i in range(len(issues)):
			# extract every issue's title
			row = df_test.loc[i, 'title']
			# apply rule-based algorithms
			single_issue_predictions = []
			if "feature request" in row.lower(): single_issue_predictions.append("Feature") 
			if "c++" in row.lower(): single_issue_predictions.append("C++")
			tokens = self.tokenize(row)
			for k, v in self.keywords.items():
				for keyword in v:
					if keyword in tokens:
						single_issue_predictions.append(k)
			rule_based_predictions.append(single_issue_predictions)
		return rule_based_predictions


	def ml_predict(self, issues, threshold=0.3):
		"""
		This method applies machine learning algorithms to predict labels
		Args: 
			issues(list): a list of issue numbers
			threshold(float): threshold of probability
		Return:
			ml_predictions(list of lists): predictions 
		"""
		# step1: fetch data
		DF = DataFetcher()
		df_test = DF.fetch_issues(issues)
		# step2: data cleaning
		SP = SentenceParser()
		SP.data = df_test
		SP.clean_body('body', True, True)
		SP.merge_column(['title', 'title', 'title', 'body'], 'train')
		test_text=SP.process_text('train', True, False, True)
		# step3: word embedding
		test_data_tfidf = self.tv.transform(test_text).toarray()
		le = LabelEncoder()
		le.fit_transform(self.labels)
		# step4: classification
		probs = self.clf.predict_proba(test_data_tfidf)
		# pick up top 2 predictions which exceeds threshold
		best_n = np.argsort(probs, axis=1)[:, -2:]
		ml_predictions=[]
		for i in range(len(best_n)):
			# INFO:Predictor:issue:11919,Performance:0.47353076240017744,Question:0.2440056213336274
			logging.info("issue:{}, {}:{}, {}:{}".format(str(issues[i]), str(le.classes_[best_n[i][-1]]), str(probs[i][best_n[i][-1]]),
						str(le.classes_[best_n[i][-2]]), str(probs[i][best_n[i][-2]])))
			single_issue_predictions = [le.classes_[best_n[i][j]]  for j in range(-1, -3, -1) if probs[i][best_n[i][j]] > threshold]
			ml_predictions.append(single_issue_predictions)
		return ml_predictions


	def predict(self, issues):
		# return predictions of both rule_base algorithms and machine learning methods
		rule_based_predictions = self.rule_based(issues)
		ml_predictions = self.ml_predict(issues)
		predictions = [list(set(rule_based_predictions[i]+ml_predictions[i])) for i in range(len(ml_predictions))]
		return predictions

		


