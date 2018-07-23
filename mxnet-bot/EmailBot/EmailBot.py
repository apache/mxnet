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
from botocore.exceptions import ClientError
from botocore.vendored import requests
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import boto3
import boto3.s3
import datetime
import logging
import os
import re

logging.basicConfig(level=logging.INFO)
logging.getLogger('boto3').setLevel(logging.CRITICAL)
logging.getLogger('botocore').setLevel(logging.CRITICAL)

LOGGER = logging.getLogger(__name__)


class EmailBot:

    # GitHub account credentials
    GITHUB_USER = os.environ.get("GITHUB_USER")
    GITHUB_OAUTH_TOKEN = os.environ.get("GITHUB_OAUTH_TOKEN")
    AUTH = (GITHUB_USER, GITHUB_OAUTH_TOKEN)
    REPO = os.environ.get("REPO")
    # Send emails using Amazon Simple Email Service(SES)
    # Sender's email address must be verified.
    # Replace it with sender's email address
    sender = "sender@email.com"
    # Recipients' email address must be verified
    # Replace it with recipients' email addresses
    recipients = ["recipient1@email.com", "recipient2@email.com"]
    # If necessary, replace us-west-2 with the AWS Region you're using for Amazon SES.
    aws_region = "us-west-2"
    # sla
    sla = 5

    def __init__(self):
        self.opendata = None
        self.closeddata = None
        # Because GitHub use UTC time, so we set self.end 2 days after today's date
        # For example:
        # self.today = "2018-07-10 00:00:00"
        # self.end = "2018-07-12 00:00:00"
        # self.start = "2018-07-04 00:00:00"
        self.start = datetime.datetime.strptime("2015-01-01", "%Y-%m-%d")
        self.end = datetime.datetime.today()+datetime.timedelta(days=2)

    def __clean_string(self, raw_string, sub_string):
        # covert all non-alphanumeric characters from raw_string to sub_string
        cleans = re.sub("[^0-9a-zA-Z]", sub_string, raw_string)
        return cleans.lower()

    def __set_period(self, period):
        # set the time period, ex: set_period(7)
        today = datetime.datetime.strptime(str(datetime.datetime.today().date()), "%Y-%m-%d")
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
        return int(self.__clean_string(response.headers['link'], " ").split()[-3])

    def read_repo(self, periodically=True):
        LOGGER.info("Start reading {} issues".format("periodically" if periodically else "all"))
        # if periodically == True, it will read issues which are created in a specific time period
        # if periodically == False, it will read all issues
        if periodically:
            self.__set_period(8)
        else:
            # 2018-5-15 is the date that 'sla' concept was used.
            self.start = datetime.datetime.strptime("2018-05-15", "%Y-%m-%d")
            self.end = datetime.datetime.today()+datetime.timedelta(days=2)
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
        assert self.opendata, "No open issues in this time period!"
        items = self.opendata
        labelled = []
        labelled_urls = ""
        unlabelled = []
        unlabelled_urls = ""
        non_responded = []
        non_responded_urls = ""
        outside_sla = []
        outside_sla_urls = ""
        responded = []
        responded_urls = ""
        total_deltas = []

        for item in items:
            url = "<a href='" + item['html_url'] + "'>" + str(item['number']) + "</a>   "
            created = datetime.datetime.strptime(item['created_at'], "%Y-%m-%dT%H:%M:%SZ")
            if item['labels']:
                labelled += [{k: v for k, v in item.items()
                              if k in ['number', 'html_url', 'title']}]
                labelled_urls = labelled_urls + url
            else:
                unlabelled += [{k: v for k, v in item.items()
                                if k in ['number', 'html_url', 'title']}]
                unlabelled_urls = unlabelled_urls + url
            if item['comments'] == 0:
                non_responded += [{k: v for k, v in item.items()
                                   if k in ['number', 'html_url', 'title']}]
                non_responded_urls = non_responded_urls + url
                if (datetime.datetime.strptime("2018-05-15", "%Y-%m-%d") < created
                        < datetime.datetime.now() - datetime.timedelta(days=self.sla)):
                    outside_sla += [{k: v for k, v in item.items()
                                     if k in ['number', 'html_url', 'title']}]
                    outside_sla_urls = outside_sla_urls + url
            else:
                responded += [{k: v for k, v in item.items()
                               if k in ['number', 'html_url', 'title']}]
                responded_urls = responded_urls + url
                comments_url = item['comments_url']
                comments = requests.get(comments_url, auth=self.AUTH)
                first_comment_created = datetime.datetime.strptime(comments.json()[0]['created_at'],
                                                                   "%Y-%m-%dT%H:%M:%SZ")
                delta = first_comment_created - created
                total_deltas.append(delta)

        data = {"labelled": labelled,
                "labelled_urls": labelled_urls,
                "unlabelled": unlabelled,
                "unlabelled_urls": unlabelled_urls,
                "responded": responded,
                "responded_urls": responded_urls,
                "non_responded": non_responded,
                "non_responded_urls": non_responded_urls,
                "outside_sla": outside_sla,
                "outside_sla_urls": outside_sla_urls,
                "total_deltas": total_deltas}

        return data

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
        self.read_repo(False)
        all_sorted_open_data = self.sort()
        self.read_repo(True)
        weekly_sorted_open_data = self.sort()
        total_deltas = weekly_sorted_open_data["total_deltas"]
        if len(total_deltas) != 0:
            avg = sum(total_deltas, datetime.timedelta())/len(total_deltas)
            avg_time = str(avg.days)+" days, "+str(int(avg.seconds/3600))+" hours"
            worst_time = str(max(total_deltas).days)+" days, "+str(int(max(total_deltas).seconds/3600)) + " hours"
        else:
            avg_time = "N/A"
            worst_time = "N/A"
        htmltable = [
                    ["Labeled issues:", str(len(weekly_sorted_open_data["labelled"]))],
                    ["Unlabeled issues:", str(len(weekly_sorted_open_data["unlabelled"]))],
                    ["List unlabeled issues", weekly_sorted_open_data["unlabelled_urls"]],
                    ["Issues with response:", str(len(weekly_sorted_open_data["responded"]))],
                    ["Issues without response:", str(len(weekly_sorted_open_data["non_responded"]))],
                    ["The average response time is:", avg_time],
                    ["The worst response time is:", worst_time],
                    ["List issues without response:", weekly_sorted_open_data["non_responded_urls"]],
                    ["Count of issues without response within 5 days:", str(len(all_sorted_open_data["outside_sla"]))],
                    ["List issues without response with 5 days:", all_sorted_open_data["outside_sla_urls"]]]
        body_html = """<html>
        <head>
        </head>
        <body>
          <h4>Week: {} to {}</h4>
          <p>{} newly issues were opened in the above period, among which {} were closed and {} are still open.</p>
          <div>{}</div>
        </body>
        </html>
                    """.format(str(self.start.date()), str((self.end - datetime.timedelta(days=2)).date()),
                               str(len(self.opendata) + len(self.closeddata)),
                               str(len(self.closeddata)), str(len(self.opendata)),
                               "\n".join(self.__html_table(htmltable)))
        return body_html

    def sendemail(self):
        # Sender's email address must be verified.
        sender = self.sender
        # If your account is still in the sandbox, this address must be verified.
        recipients = self.recipients
        # If necessary, replace us-west-2 with the AWS Region you're using for Amazon SES.
        aws_region = self.aws_region
        # The email body for recipients with non-HTML email clients.
        body_text = "weekly report"
        # The HTML body of the email.
        body_html = self.__bodyhtml()
        # The subject line for the email.
        subject = "GitHub Issues Daily Report {} to {}".format(str(self.start.date()),
                                                               str((self.end - datetime.timedelta(days=2)).date()))
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
        msg.attach(msg_body)

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
