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
LOGGER = logging.getLogger(__name__)


def train_general_labels():
    # This function is to train issues with general labels
    # Word Embedding model: TFIDF, trigram, max_features=10000
    # Classifier: SVC, kernel = 'rbf'
    LOGGER.info("Start training issues of general labels")
    target_labels = ["Performance", "Test", "Question",
                     "Feature request", "Call for contribution",
                     "Feature", "Example", "Doc",
                     "Installation", "Build", "Bug"]
    # Step1: Fetch issues with general labels
    LOGGER.info("Fetching Data..")
    DF = DataFetcher()
    filename = DF.data2json('all', target_labels, False)
    # Step2: Clean data
    LOGGER.info("Cleaning Data..")
    SP = SentenceParser()
    SP.read_file(filename, 'json')
    SP.clean_body('body', True, True)
    SP.merge_column(['title', 'title', 'title', 'body'], 'train')
    text = SP.process_text('train', True, False, True)
    df = SP.data
    # Step3: Word Embedding
    LOGGER.info("Word Embedding..")
    # TFIDF model, trigram, max_features= 10000
    tv = TfidfVectorizer(min_df=0.00009, ngram_range=(1, 3), max_features=10000)
    X = tv.fit_transform(text).toarray()
    # Labels
    labels = SP.data['labels']
    le = LabelEncoder()
    Y = le.fit_transform(labels)
    # Step4: Train Classifier
    # SVC, kernel = 'rbf'
    LOGGER.info("Training Data..")
    clf = SVC(gamma=0.5, C=100, probability=True)
    clf.fit(X, Y)
    # Step5: save models
    LOGGER.info("Saving Models..")
    pickle.dump(tv, open("Vectorizer.p", "wb"))
    pickle.dump(clf, open("Classifier.p", "wb"))
    pickle.dump(labels, open("Labels.p", "wb"))
    LOGGER.info("Completed!")
    return


def train_operator_label():
    # This function is to train issues with 'Operator' label
    # Word Embedding model: TFIDF, bigram, max_features=10000
    # Classifier: Gaussian Naive Bayes
    LOGGER.info("Start training issues of Operator label")
    # Step1: Fetch all labeled issues
    LOGGER.info("Fetching Data..")
    DF = DataFetcher()
    filename = DF.data2json(state='all', labels=None)
    # Step2: Clean data
    LOGGER.info("Cleaning Data..")
    SP = SentenceParser()
    SP.read_file(filename, 'json')
    df = SP.data
    # divide data into 2 parts: issues have 'Operator' label vs issues don't have
    df['clf'] = 'False'
    for i in range(0, len(SP.data)):
        if 'Operator' in df['labels'][i]:
            df.loc[i, 'clf'] = "True"
    SP.clean_body('body', True, True)
    SP.merge_column(['title', 'title', 'title', 'body'], 'train')
    # Step3: Word Embedding
    LOGGER.info("Word Embedding..")
    text = SP.process_text('train', True, False, True)
    tv = TfidfVectorizer(min_df=0.00009, ngram_range=(1, 2), max_features=10000)
    X = tv.fit_transform(text).toarray()
    labels = df['clf']
    le = LabelEncoder()
    Y = le.fit_transform(labels)
    # Step4: Train Classifier
    LOGGER.info("Training Data..")
    clfg = GaussianNB()
    clfg.fit(X, Y)
    # Step 5: save models
    LOGGER.info("Saving Models..")
    pickle.dump(tv, open("Vectorizer_Operator.p", "wb"))
    pickle.dump(clfg, open("Classifier_Gaussian.p", "wb"))
    LOGGER.info("Completed!")
    return




