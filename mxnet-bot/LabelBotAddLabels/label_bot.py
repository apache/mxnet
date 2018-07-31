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

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.getLogger('boto3').setLevel(logging.CRITICAL)
logging.getLogger('botocore').setLevel(logging.CRITICAL)

# Comment: "@mxnet-label-bot, please add labels: bug, test"
# Then, this script will recognize this comment and add labels
secret = json.loads(secret_manager.get_secret())
GITHUB_USER = secret["GITHUB_USER"]
GITHUB_OAUTH_TOKEN = secret["GITHUB_OAUTH_TOKEN"]
REPO = os.environ.get("REPO")
AUTH = (GITHUB_USER, GITHUB_OAUTH_TOKEN)


def tokenize(string):
    substring = string[string.find('[')+1: string.rfind(']')] 
    labels = [' '.join(label.split()) for label in substring.split(',')]
    logger.info("recognize labels: {}".format(", ".join(labels)))
    return labels


def clean_string(raw_string, sub_string):
    # covert all non-alphanumeric characters from raw_string to sub_string
    cleans = re.sub("[^0-9a-zA-Z]", sub_string, raw_string)
    return cleans.lower()


def count_pages(obj, state='all'):
    # This method is to count how many pages of issues/labels in total
    # obj could be "issues"/"labels"
    # state could be "open"/"closed"/"all", available to issues
    assert obj in set(["issues", "labels"]), "Invalid Input!"
    url = 'https://api.github.com/repos/{}/{}'.format(REPO, obj)
    if obj == 'issues':
        response = requests.get(url, {'state': state},
                                auth=AUTH)
    else:
        response = requests.get(url, auth=AUTH)
    assert response.status_code == 200, response.status_code
    if "link" not in response.headers:
        return 1
    # response.headers['link'] will looks like:
    # <https://api.github.com/repositories/34864402/issues?state=all&page=387>; rel="last"
    # In this case we need to extrac '387' as the count of pages
    return int(clean_string(response.headers['link'], " ").split()[-3])


def find_notifications():
    data = []
    pages = count_pages("issues")
    for page in range(1, pages+1):
        url = 'https://api.github.com/repos/' + REPO + '/issues?page=' + str(page) \
            + '&per_page=30'.format(repo=REPO)
        response = requests.get(url,
                                {'state': 'open',
                                 'base': 'master',
                                 'sort': 'created',
                                 'direction': 'desc'},
                                auth=AUTH)
        for item in response.json():
            # limit the amount of unlabeled issues per execution
            if len(data) >= 50:
                break
            if "pull_request" in item:
                continue
            if not item['labels']:
                if item['comments'] != 0:
                    labels = []
                    comments_url = "https://api.github.com/repos/{}/issues/{}/comments".format(REPO,item['number'])
                    comments = requests.get(comments_url, auth=AUTH).json()
                    for comment in comments:
                        if "@mxnet-label-bot" in comment['body']:
                            labels += tokenize(comment['body'])
                            logger.info("issue: {}, comment: {}".format(str(item['number']),comment['body']))
                    if labels != []:
                        data.append({"issue": item['number'], "labels": labels})
    return data


def all_labels():
    pages = count_pages("labels")
    all_labels = []
    for page in range(1, pages+1):
        url = 'https://api.github.com/repos/' + REPO + '/labels?page=' + str(page) \
            + '&per_page=30'.format(repo=REPO)
        response = requests.get(url, auth=AUTH)
        for item in response.json():
            all_labels.append(item['name'].lower())
    logger.info("{} labels in total".format(str(len(all_labels))))
    return set(all_labels)


all_labels = all_labels()


def add_github_labels(number, labels):
    # number: the issue number
    # labels: list of strings
    issue_labels_url = 'https://api.github.com/repos/{repo}/issues/{id}/labels'\
                        .format(repo=REPO, id=number)
    # clean labels, remove duplicated spaces. ex: "hello  world" -> "hello world"
    labels = [" ".join(label.split()) for label in labels]
    labels = [label for label in labels if label.lower() in all_labels]
    response = requests.post(issue_labels_url, json.dumps(labels), auth=AUTH)
    if response.status_code == 200:
        logger.info('Successfully add labels to {}: {}.'.format(str(number), str(labels)))
    else:
        logger.error("Could not add the label")
        logger.error(response.json())


def label(issues):
    #issues is a json file: [{number:1, labels:[a,b]},{number:1, labels:[a,b]}]
    for issue in issues:
        add_github_labels(issue['issue'], issue['labels'])


