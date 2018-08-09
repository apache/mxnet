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

# This scipt is served to fetch GitHub issues into a json file
from __future__ import print_function
import os
import requests
import json
import re
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)


class DataFetcher:

    def __init__(self,
                 github_user = os.environ.get("github_user"),
                 github_oauth_token = os.environ.get("github_oauth_token"),
                 repo = os.environ.get("repo")):
        """
        This DataFetcher serves to fetch issues data
        Args:
            github_user(str): the github id. ie: "CathyZhang0822"
            github_oauth_token(str): the github oauth token, paired with github_user to realize authorization
            repo(str): the repo name
        """
        self.github_user = github_user
        self.github_oauth_token = github_oauth_token
        self.repo = repo
        self.auth = (self.github_user, self.github_oauth_token)
        self.json_data = None

    def cleanstr(self, raw_string, sub_string):
        """
        This method is to convert all non-alphanumeric charaters from 
        raw_string into substring
        """
        clean = re.sub("[^0-9a-zA-Z]", sub_string, raw_string)
        return clean.lower()

    def count_pages(self, state):
        """
        This method is to count how many pages of issues/labels in total
        state can be "open"/"closed"/"all"
        """
        url = 'https://api.github.com/repos/%s/issues' % self.repo
        response = requests.get(url, {'state': state},
                                auth=self.auth)
        assert response.status_code == 200, "Authorization failed"
        if "link" not in response.headers:
            return 1
        return int(self.cleanstr(response.headers['link'], " ").split()[-3])
    
    def fetch_issues(self, issue_nums):
        """
        This method is to fetch issues data
        issue_num: a list of issue ids
        return issues' data in pandas dataframe format
        """
        assert issue_nums != [], "Empty Input!"
        logging.info("Reading issues:{}".format(", ".join([str(num) for num in issue_nums])))
        data = []
        for number in issue_nums:
            url = 'https://api.github.com/repos/' + self.repo + '/issues/' + str(number)
            response = requests.get(url, auth=self.auth)
            item = response.json()
            assert 'title' in item, "{} issues doesn't exist!".format(str(number))
            data += [{'id': str(number),'title': item['title'], 'body': item['body']}]
        return pd.DataFrame(data)

    def data2json(self,state,labels=None, other_labels = False):
        """
        This method is to store issues' data into a json file, return json file's name
        state can be either "open"/"closed"/"all"
        labels is a list of target labels we are interested in
        other_labels can be either "True"/"False"
        """
        assert state in set(['all', 'open', 'closed']), "Invalid State!"
        logging.info("Reading {} issues..".format(state))
        pages = self.count_pages(state)
        data = []
        for x in range(1, pages+1):
            url = 'https://api.github.com/repos/' + self.repo + '/issues?page=' + str(x) \
                  + '&per_page=30'.format(repo=self.repo)
            response = requests.get(url,
                                    {'state':state,
                                     'base':'master',
                                     'sort':'created'},
                                     auth=self.auth)
            for item in response.json():
                if "pull_request" in item:
                    continue
                if "labels" in item:
                    issue_labels=list(set([item['labels'][i]['name'] for i in range(len(item['labels']))]))
                else:
                    continue
                if labels!= None:
                    # fetch issue which has at least one target label
                    for label in labels:
                        if label in issue_labels:
                            if other_labels:
                                # besides target labels, we still want other labels
                                data += [{'id': item['number'],'title': item['title'], 'body': item['body'], 'labels': issue_labels}]
                            else:
                                # only record target labels
                                if(label in set(["Feature", "Call for Contribution", "Feature request"])):
                                    label = "Feature"
                                data += [{'id': item['number'],'title': item['title'], 'body': item['body'], 'labels': label}]
                            # if have this break, then we only pick up the first target label
                            break
                else:
                    # fetch all issues
                    data += [{'id': item['number'],'title': item['title'], 'body': item['body'], 'labels': issue_labels}]                                      
        self.json_data = data
        s_labels = "_".join(labels) if labels!=None else "all_labels"
        filename = "{}_data.json_{}".format(state,s_labels)
        logging.info("Writing json file..")
        with open(filename,'w') as write_file:
            json.dump(data, write_file)
        logging.info("{} json file is ready!".format(filename))
        return filename
