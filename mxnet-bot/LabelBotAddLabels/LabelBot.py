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
import json
import os
from botocore.vendored import requests
import re
import logging
import secret_manager
logging.basicConfig(level=logging.INFO)

class LabelBot:

    def __init__(self, 
                 repo=os.environ.get("repo"), 
                 github_user=None, 
                 github_oauth_token=None, 
                 secret=True):
        self.repo = repo
        self.github_user = github_user
        self.github_oauth_token = github_oauth_token
        if secret:
            self.get_secret()
        self.auth = (self.github_user, self.github_oauth_token)
        self.all_labels = None

    def get_secret(self):
        """
        This method is to get secret value from Secrets Manager
        """
        secret = json.loads(secret_manager.get_secret())
        self.github_user = secret["github_user"]
        self.github_oauth_token = secret["github_oauth_token"]

    def tokenize(self, string):
        """
        This method is to extract labels from comments
        """
        substring = string[string.find('[')+1: string.rfind(']')] 
        labels = [' '.join(label.split()) for label in substring.split(',')]
        return labels

    def clean_string(self, raw_string, sub_string):
        """
        This method is to convert all non-alphanumeric characters from raw_string to sub_string
        """
        cleans = re.sub("[^0-9a-zA-Z]", sub_string, raw_string)
        return cleans.lower()

    def count_pages(self, obj, state='all'):
        """
        This method is to count how many pages of issues/labels in total
        obj could be "issues"/"labels"
        state could be "open"/"closed"/"all", available to issues
        """
        assert obj in set(["issues", "labels"]), "Invalid Input!"
        url = 'https://api.github.com/repos/{}/{}'.format(self.repo, obj)
        if obj == 'issues':
            response = requests.get(url, {'state': state}, auth=self.auth)    
        else:
            response = requests.get(url, auth=self.auth)
        assert response.status_code == 200, response.status_code
        if "link" not in response.headers:
            return 1
        return int(self.clean_string(response.headers['link'], " ").split()[-3])

    def find_notifications(self):
        """
        This method is to find comments which @mxnet-label-bot
        """
        data = []
        pages = self.count_pages("issues")
        for page in range(1, pages+1):
            url = 'https://api.github.com/repos/' + self.repo + '/issues?page=' + str(page) \
                + '&per_page=30'.format(repo=self.repo)
            response = requests.get(url,
                                    {'state': 'open',
                                     'base': 'master',
                                     'sort': 'created',
                                     'direction': 'desc'},
                                     auth=self.auth)
            for item in response.json():
                # limit the amount of unlabeled issues per execution
                if len(data) >= 50:
                    break
                if "pull_request" in item:
                    continue
                if not item['labels']:
                    if item['comments'] != 0:
                        labels = []
                        comments_url = "https://api.github.com/repos/{}/issues/{}/comments".format(self.repo,item['number'])
                        comments = requests.get(comments_url, auth=self.auth).json()
                        for comment in comments:
                            if "@mxnet-label-bot" in comment['body']:
                                labels += self.tokenize(comment['body'])
                                logging.info("issue: {}, comment: {}".format(str(item['number']),comment['body']))
                        if labels != []:
                            data.append({"issue": item['number'], "labels": labels})
        return data

    def find_all_labels(self):
        """
        This method is to find all existing labels in the repo
        """
        pages = self.count_pages("labels")
        all_labels = []
        for page in range(1, pages+1):
            url = 'https://api.github.com/repos/' + self.repo + '/labels?page=' + str(page) \
                + '&per_page=30'.format(repo=self.repo)
            response = requests.get(url, auth=self.auth)
            for item in response.json():
                all_labels.append(item['name'].lower())
        self.all_labels = set(all_labels)
        return set(all_labels)

    def add_github_labels(self, issue_num, labels):
        """
        This method is to add a list of labels to one issue.
        First it will remove redundant white spaces from each label.
        Then it will check whether labels exist in the repo.
        At last, it will add existing labels to the issue
        """
        assert self.all_labels, "Find all labels first"
        issue_labels_url = 'https://api.github.com/repos/{repo}/issues/{id}/labels'\
                            .format(repo=self.repo, id=issue_num)
        # clean labels, remove duplicated spaces. ex: "hello  world" -> "hello world"
        labels = [" ".join(label.split()) for label in labels]
        labels = [label for label in labels if label.lower() in self.all_labels]
        response = requests.post(issue_labels_url, json.dumps(labels), auth=self.auth)
        if response.status_code == 200:
            logging.info('Successfully add labels to {}: {}.'.format(str(issue_num), str(labels)))
        else:
            logging.error("Could not add the label")
            logging.error(response.json())

    def label(self, issues):
        """
        This method is to add labels to multiple issues
        Input is a json file: [{number:1, labels:[a,b]},{number:2, labels:[c,d]}]
        """
        self.find_all_labels()
        for issue in issues:
            self.add_github_labels(issue['issue'], issue['labels'])

