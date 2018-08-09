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

# This script is served to train Machine Learning models
from DataFetcher import DataFetcher
from SentenceParser import SentenceParser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
import pickle
import logging
logging.basicConfig(level=logging.INFO)


class Trainer:

    def __init__(self):
        """
        Trainer is to train issues using Machine Learning methods.
        self.labels(list): a list of target labels
        self.tv: TFIDF model (trigram, max_features = 10000)
        self.clf: Classifier (SVC, kenerl = 'rbf')
        """
        self.labels = ["Performance", "Test", "Question",
                       "Feature request", "Call for contribution",
                       "Feature", "Example", "Doc",
                       "Installation", "Build", "Bug"]
        self.tv = TfidfVectorizer(min_df=0.00009, ngram_range=(1, 3), max_features=10000)
        self.clf = SVC(gamma=0.5, C=100, probability=True)

    def train(self):
        """
        This method is to train and save models.
        """
        logging.info("Start training issues of general labels")
        # Step1: Fetch issues with general labels
        logging.info("Fetching Data..")
        DF = DataFetcher()
        filename = DF.data2json('all', self.labels, False)
        # Step2: Clean data
        logging.info("Cleaning Data..")
        SP = SentenceParser()
        SP.read_file(filename, 'json')
        SP.clean_body('body', True, True)
        SP.merge_column(['title', 'title', 'title', 'body'], 'train')
        text = SP.process_text('train', True, False, True)
        df = SP.data
        # Step3: Word Embedding
        logging.info("Word Embedding..")
        # tv = TfidfVectorizer(min_df=0.00009, ngram_range=(1, 3), max_features=10000)
        tv = self.tv
        X = tv.fit_transform(text).toarray()
        # Labels
        labels = SP.data['labels']
        le = LabelEncoder()
        Y = le.fit_transform(labels)
        # Step4: Train Classifier
        # SVC, kernel = 'rbf'
        logging.info("Training Data..")
        #clf = SVC(gamma=0.5, C=100, probability=True)
        clf = self.clf
        clf.fit(X, Y)
        # Step5: save models
        logging.info("Saving Models..")
        pickle.dump(tv, open("/tmp/Vectorizer.p", "wb"))
        pickle.dump(clf, open("/tmp/Classifier.p", "wb"))
        pickle.dump(labels, open("/tmp/Labels.p", "wb"))
        logging.info("Completed!")
        return





