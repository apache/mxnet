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

from sklearn.preprocessing import LabelEncoder
from SentenceParser import SentenceParser
from DataFetche import DataFetcher
import numpy as np
import pickle
import re
import logging

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

def tokenize(row):
    row = re.sub('[^a-zA-Z0-9]', ' ', row).lower()
    words = set(row.split())
    return words

def rule_based(row):
    # return a list of rule_based labels
    l = []
    # 'feature request' in the issue's title
    if "feature request" in row.lower():
        l.append("Feature")
    # 'c++' in the issue's title
    if "c++" in row.lower():
        l.append("C++")
    tokens = tokenize(row)
    ci_keywords = ["ci", "ccache", "jenkins"]
    flaky_keywords = ["flaky"]
    gluon_keywords = ["gluon"]
    cuda_keywords = ["cuda", "cudnn"]
    scala_keywords = ["scala"]
    # one of ci keywords in the issue's title
    for key in ci_keywords:
        if key in tokens:
            l.append("CI")
            break
    # one of flaky keywords in the issue's title
    for key in flaky_keywords:
        if key in tokens:
            l.append("Flaky")
    # one of gluon keywords in the issue's title
    for key in gluon_keywords:
        if key in tokens:
            l.append("Gluon")
    # one of scala keywords in the issue's title
    for key in scala_keywords:
        if key in tokens:
            l.append("Scala")
    # one of cudo keywords in the issue's title
    for key in cuda_keywords:
        if key in tokens:
            l.append("CUDA")
    return l

def predict(issues):
    # get Machine Learning models' predictions
    # return Rule-based predictions and Machine Learning models' predictions together
    # step1: fetch data
    DF = DataFetcher()
    df_test = DF.fetch_issues(issues)
    # step2: data cleaning
    SP = SentenceParser()
    SP.data = df_test
    SP.clean_body('body', True, True)
    SP.merge_column(['title', 'title', 'title', 'body'], 'train')
    test_text = SP.process_text('train', True, False, True)
    # step3: word embedding
    tv = pickle.load(open("Vectorizer.p", "rb"))
    test_data_tfidf = tv.transform(test_text).toarray()
    tv_op = pickle.load(open("Vectorizer_Operator.p", "rb"))
    test_data_tfidf_operator = tv_op.transform(test_text).toarray()
    labels = pickle.load(open("Labels.p", "rb"))
    le = LabelEncoder()
    le.fit_transform(labels)
    # step4: classification
    clf = pickle.load(open("Classifier.p", "rb"))
    probs = clf.predict_proba(test_data_tfidf)
    # pickup top 2 predictions which exceeds threshold 0.3
    best_n = np.argsort(probs, axis=1)[:, -2:]
    clf_g = pickle.load(open("Classifier_Gaussian.p", "rb"))
    ops_pre = clf_g.predict(test_data_tfidf_operator)
    recommendations = []
    for i in range(len(best_n)):
        l = rule_based(df_test.loc[i, 'title'])
        l += [le.classes_[best_n[i][j]]  for j in range(-1, -3, -1) if probs[i][best_n[i][j]] > 0.3]
        l += ["Operator"] if ops_pre[i] == 1 else ""
        recommendations.append(l)
        LOGGER.info(str(issues[i]), str(le.classes_[best_n[i][-1]]), probs[i][best_n[i][-1]], str(le.classes_[best_n[i][-2]]), probs[i][best_n[i][-2]])
    return recommendations
