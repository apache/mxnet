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
logging.getLogger('boto3').setLevel(logging.CRITICAL)
logging.getLogger('botocore').setLevel(logging.CRITICAL)

LOGGER = logging.getLogger(__name__)


class DataFetcher:
    GITHUB_USER = os.environ.get("GITHUB_USER")
    GITHUB_OAUTH_TOKEN = os.environ.get("GITHUB_OAUTH_TOKEN")
    REPO = os.environ.get("REPO")
    assert GITHUB_USER and GITHUB_OAUTH_TOKEN and REPO, "Please set environment variables!"
    REPO_URL = 'https://api.github.com/repos/%s/issues' % REPO
    AUTH = (GITHUB_USER, GITHUB_OAUTH_TOKEN)

    def __init__(self):
        self.json_data = None

    def cleanstr(self, s, e):
        # convert all non-alphanumeric charaters into e
        cleanstr = re.sub("[^0-9a-zA-Z]", e, s)
        return cleanstr.lower()

    def count_pages(self, state):
        # This method is to count how many pages of issues/labels in total
        # state could be "open"/"closed"/"all", available to issues
        response = requests.get(self.REPO_URL, {'state': state},
                                auth=self.AUTH)
        assert response.headers["Status"] == "200 OK", "Authorization failed"
        if "link" not in response.headers:
            # That means only 1 page exits
            return 1
        # response.headers['link'] will looks like:
        # <https://api.github.com/repositories/34864402/issues?state=all&page=387>; rel="last"
        # In this case we need to extract '387' as the count of pages
        # return the number of pages
        return int(self.cleanstr(response.headers['link'], " ").split()[-3])
    
    def fetch_issues(self, numbers):
        # number: a list of issue ids
        # return issues' data in pandas dataframe format
        assert numbers != [], "Empty Input!"
        LOGGER.info("Reading issues:{}".format(", ".join([str(num) for num in numbers])))
        data = []
        for number in numbers:
            url = 'https://api.github.com/repos/' + self.REPO + '/issues/' + str(number)
            response = requests.get(url, auth=self.AUTH)
            item = response.json()
            assert 'title' in item, "{} issues doesn't exist!".format(str(number))
            data += [{'id': str(number),'title': item['title'], 'body': item['body']}]
        return pd.DataFrame(data)

    def data2json(self,state,labels=None, other_labels = False):
        # store issues' data into a json file, return json file's name
        # state can be either "open"/"closed"/"all"
        # labels is a list of target labels we are interested int
        # other_labels can be either "True"/"False"
        assert state in set(['all', 'open', 'closed']), "Invalid State!"
        LOGGER.info("Reading {} issues..".format(state))
        pages = self.count_pages(state)
        data = []
        for x in range(1, pages+1):
            url = 'https://api.github.com/repos/' + self.REPO + '/issues?page=' + str(x) \
                  + '&per_page=30'.format(repo=self.REPO)
            response = requests.get(url,
                                    {'state':state,
                                     'base':'master',
                                     'sort':'created'},
                                     auth=self.AUTH)
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
        LOGGER.info("Writing json file..")
        with open(filename,'w') as write_file:
            json.dump(data, write_file)
        LOGGER.info("{} json file is ready!".format(filename))
        return filename









