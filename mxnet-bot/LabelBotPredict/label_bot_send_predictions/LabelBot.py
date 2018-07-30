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

from __future__ import print_function
import datetime
import operator
import os
from collections import defaultdict
import boto3
from botocore.vendored import requests
import re
from botocore.exceptions import ClientError
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.text import MIMEText
import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger('boto3').setLevel(logging.CRITICAL)
logging.getLogger('botocore').setLevel(logging.CRITICAL)

LOGGER = logging.getLogger(__name__)

"""
from LabelBot import LabelBot as LB
lb = LB()
lb.sendemail()
"""


class LabelBot:

    # fetch environment variables
    GITHUB_USER = os.environ.get("GITHUB_USER")
    GITHUB_OAUTH_TOKEN = os.environ.get("GITHUB_OAUTH_TOKEN")
    REPO = os.environ.get("REPO")
    AUTH = (GITHUB_USER, GITHUB_OAUTH_TOKEN)
    # Sender's email address must be verified. 
    # ie: SENDER = "a@gmail.com"
    sender = os.environ.get("SENDER")
    # Recipients' email address must be verified
    # ie: RECIPIENTS = "a@gmail.com, b@gmail.com"
    recipients = [s.strip() for s in os.environ.get("RECIPIENTS").split(",")]
    # If necessary, replace us-west-2 with the AWS Region you're using for Amazon SES.
    # ie: AWS_REGION = "us-west-2"
    aws_region = os.environ.get('AWS_REGION')
    # Elastic Beanstalk Server URL
    # ie: EB_URL = "http://cathydocker-env.63rbye2pys.us-west-2.elasticbeanstalk.com"
    elastic_beanstalk_url = os.environ.get("EB_URL")

    def __init__(self):
        self.opendata = None
        self.closeddata = None
        self.sorted_open_issues = None
        self.start = datetime.datetime.strptime("2015-01-01", "%Y-%m-%d")
        self.end = datetime.datetime.today()

    def __clean_string(self, raw_string, sub_string):
        # covert all non-alphanumeric characters from raw_string to sub_string
        cleans = re.sub("[^0-9a-zA-Z]", sub_string, raw_string)
        return cleans.lower()

    def __set_period(self, period):
        # set the time period, ex: set_period(7)
        today = datetime.datetime.strptime(str(datetime.datetime.today().date()), "%Y-%m-%d")
        # Because GitHub use UTC time, so we set self.end 2 days after today's date
        # For example:
        # self.today = "2018-07-10 00:00:00"
        # self.end = "2018-07-12 00:00:00"
        # self.start = "2018-07-04 00:00:00"
        self.end = today + datetime.timedelta(days=2)
        timedelta = datetime.timedelta(days=period)
        self.start = self.end - timedelta

    def __count_pages(self, obj, state='all'):
        # This method is to count how many pages of issues/labels in total
        # obj could be "issues"/"labels"
        # state could be "open"/"closed"/"all", available to issues
        assert obj in set(["issues", "labels"]), "Invalid Input!"
        url = 'https://api.github.com/repos/{}/{}'.format(self.REPO, obj)
        if obj == 'issues':
            response = requests.get(url, {'state': state},
                                    auth=self.AUTH)
        else:
            response = requests.get(url, auth=self.AUTH)
        assert response.headers["Status"] == "200 OK", response.headers["Status"]
        if "link" not in response.headers:
            # That means only 1 page exits
            return 1
        # response.headers['link'] will looks like:
        # <https://api.github.com/repositories/34864402/issues?state=all&page=387>; rel="last"
        # In this case we need to extrac '387' as the count of pages
        # return how many pages in total
        return int(self.__clean_string(response.headers['link'], " ").split()[-3])

    def read_repo(self, periodically=True):
        LOGGER.info("Start reading repo")
        # if periodically == True, it will read issues which are created in a specific time period
        # if periodically == False, it will read all issues
        if periodically:
            self.__set_period(8)
        pages = self.__count_pages('issues', 'all')
        opendata = []
        closeddata = []
        stop = False
        for page in range(1, pages + 1):
            url = 'https://api.github.com/repos/' + self.REPO + '/issues?page=' + str(page) \
                  + '&per_page=30'.format(repo=self.REPO)
            response = requests.get(url,
                                    {'state': 'all',
                                     'base': 'master',
                                     'sort': 'created',
                                     'direction': 'desc'},
                                    auth=self.AUTH)
            response.raise_for_status()
            for item in response.json():
                if "pull_request" in item:
                    continue
                created = datetime.datetime.strptime(item['created_at'], "%Y-%m-%dT%H:%M:%SZ")
                if self.start <= created <= self.end:
                    if item['state'] == 'open':
                        opendata.append(item)
                    elif item['state'] == 'closed':
                        closeddata.append(item)
                else:
                    stop = True
                    break
            if stop:
                break
        self.opendata = opendata
        self.closeddata = closeddata

    def sort(self):
        # sort data, return a dictionary
        self.read_repo(True)
        assert self.opendata, "No open issues in this time period!"
        items = self.opendata
        labelled = []
        labelled_urls = ""
        unlabelled = []
        unlabelled_urls = ""
        labels = {}
        labels = defaultdict(lambda: 0, labels)
        non_responded = []
        non_responded_urls = ""
        responded = []
        responded_urls = ""

        for item in items:
            url = "<a href='" + item['html_url'] + "'>" + str(item['number']) + "</a>   "
            if item['labels'] != []:
                labelled += [{k: v for k, v in item.items()
                              if k in ['number', 'html_url', 'title']}]
                labelled_urls = labelled_urls + url
                for label in item['labels']:
                    labels[label['name']] += 1
            else:
                unlabelled += [{k: v for k, v in item.items()
                                if k in ['number', 'html_url', 'title']}]
                unlabelled_urls = unlabelled_urls + url
            if item['comments'] == 0:
                non_responded += [{k: v for k, v in item.items()
                                   if k in ['number', 'html_url', 'title']}]
                non_responded_urls = non_responded_urls + url
            else:
                responded += [{k: v for k, v in item.items()
                               if k in ['number', 'html_url', 'title']}]
                responded_urls = responded_urls + url
        labels['unlabelled'] = len(unlabelled)
        data = {"labelled": labelled,
                "labels": labels,
                "labelled_urls": labelled_urls,
                "unlabelled": unlabelled,
                "unlabelled_urls": unlabelled_urls,
                "non_responded": non_responded,
                "non_responded_urls": non_responded_urls,
                "responded": responded,
                "responded_urls": responded_urls}
        self.sorted_open_issues = data
        return data

    def predict(self):
        assert self.sorted_open_issues, "Please call .sort()) first"
        data = self.sorted_open_issues
        unlabeled_data_number = [item['number'] for item in data["unlabelled"]]
        LOGGER.info("Start predicting labels for: {}".format(str(unlabeled_data_number)))
        url = "{}/predict".format(self.elastic_beanstalk_url)
        response = requests.post(url, json={"issues": unlabeled_data_number})
        LOGGER.info(response.json())
        return response.json()

    def __html_table(self, lol):
        # Generate html table of lol(list of lists)
        yield '<table style="width: 500px;">'
        for sublist in lol:
            yield '  <tr><td style = "width:200px;">'
            yield '    </td><td style = "width:300px;">'.join(sublist)
            yield '  </td></tr>'
        yield '</table>'

    def __bodyhtml(self):
        # This is body html of email content
        self.sort()
        data = self.sorted_open_issues
        all_labels = data['labels']
        sorted_labels = sorted(all_labels.items(), key=operator.itemgetter(1), reverse=True)
        labels = [item[0] for item in sorted_labels[:10]]
        fracs = [item[1] for item in sorted_labels[:10]]
        url = "{}/draw".format(self.elastic_beanstalk_url)
        pic_data = {"fracs": fracs, "labels": labels}
        response = requests.post(url, json=pic_data)
        if response.status_code == 200:
            with open("/tmp/sample.png", "wb") as f:
                f.write(response.content)
        htmltable = [["Count of issues with no response:", str(len(data['non_responded']))],
                     ["List of issues with no response:", data['non_responded_urls']],
                     ["Count of unlabeled issues:", str(len(data['unlabelled']))],
                     ["List of unlabeled issues:", data['unlabelled_urls']]
                     ]
        htmltable2 = [[str(item['number']), ",".join(item['predictions'])] for item in self.predict()]
        body_html = """<html>
        <head>
        </head>
        <body>
          <h4>Week: {} to {}</h4>
          <p>{} newly issues were opened in the above period, among which {} were closed and {} are still open.</p>
          <div>{}</div>
          <p>Here are the recommanded labels for unlabeled issues:</p>
          <div>{}</div>
          <p><img src="cid:image1" width="400" height="400"></p>
        </body>
        </html>
                    """.format(str(self.start.date()), str((self.end - datetime.timedelta(days=2)).date()),
                               str(len(self.opendata) + len(self.closeddata)),
                               str(len(self.closeddata)), str(len(self.opendata)),
                               "\n".join(self.__html_table(htmltable)),
                               "\n".join(self.__html_table(htmltable2)))
        return body_html

    def sendemail(self):
        # Sender's email address must be verified.
        sender = self.sender
        # If your account is still in the sandbox, this address must be verified.
        recipients = self.recipients
        # If necessary, replace us-west-2 with the AWS Region you're using for Amazon SES.
        aws_region = self.aws_region
        # The subject line for the email.
        subject = "weekly report"
        # The email body for recipients with non-HTML email clients.
        body_text = "weekly report"
        # The HTML body of the email.
        body_html = self.__bodyhtml()
        # The character encoding for the email.
        charset = "utf-8"
        # Create a new SES resource and specify a region.
        client = boto3.client('ses', region_name=aws_region)

        # Create a multipart/mixed parent container.
        msg = MIMEMultipart('mixed')
        # Add subject, from and to lines
        msg['Subject'] = subject
        msg['From'] = sender
        msg['To'] = ",".join(recipients)

        # Create a multiparter child container
        msg_body = MIMEMultipart('alternative')

        # Encode the text and HTML content and set the character encoding. This step is
        # necessary if you're sending a message with characters outside the ASCII range.
        textpart = MIMEText(body_text.encode(charset), 'plain', charset)
        htmlpart = MIMEText(body_html.encode(charset), 'html', charset)

        # Add the text and HTML parts to the child container
        msg_body.attach(textpart)
        msg_body.attach(htmlpart)

        # Attach the multipart/alternative child container to the multipart/mixed parent container
        msg.attach(msg_body)

        # Attach Image
        fg = open('/tmp/sample.png', 'rb')
        msg_image = MIMEImage(fg.read())
        fg.close()
        msg_image.add_header('Content-ID', '<image1>')
        msg.attach(msg_image)
        try:
            # Provide the contents of the email.
            response = client.send_raw_email(
                Source=sender,
                Destinations=recipients,
                RawMessage={
                    'Data': msg.as_string(),
                },
            )
        # Display an error if something goes wrong. 
        except ClientError as e:
            LOGGER.error(e.response['Error']['Message'])
        else:
            LOGGER.info("Email sent! Message ID:"),
            LOGGER.info(response['MessageId'])
        return
