#!/usr/bin/env python

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


import sys
from subprocess import check_output, STDOUT, CalledProcessError


def prepare_link_test_result(command_output):
    # Constant string patterns
    NEW_PAGE_TEST_START_REGEX = "Getting links from:"
    BROKEN_PAGE_START_REGEX = "BROKEN"
    PAGE_TEST_END_REGEX = "Finished! "

    # Whitelisted broken links patterns
    HTTP_403_REGEX = "(HTTP_403)"
    HTTP_401_REGEX = "(HTTP_401)"
    HTTP_409_REGEX = "(HTTP_409)"
    HTTP_3XX_REGEX = "(HTTP_3"
    BLC_UNKNOWN_REGEX = "(BLC_UNKNOWN)"
    HTTP_UNDEFINED = "HTTP_undefined"
    FALSE_SCALA_API_DOC_LINK = "java$lang.html"
    FALSE_SCALA_API_DEPRECATED_LINK = "api/scala/docs/deprecated-list.html"
    FALSE_PAPER_LINK = "https://static.googleusercontent.com/media/research.google.com/en/pubs/archive/45488.pdf"

    # Initialize flags with happy case
    current_page = ""
    current_page_broken = False
    current_page_broken_links = ""

    broken_links_count = 0
    broken_links_summary = ""

    for line in command_output.splitlines():
        if line.startswith(NEW_PAGE_TEST_START_REGEX):
            # New page test is starting. Reset the flags
            current_page = line.split(NEW_PAGE_TEST_START_REGEX)[1]
            current_page_broken = False
            current_page_broken_links = ""

        if line.find(BROKEN_PAGE_START_REGEX) != -1:
            # Skip (401, 403, 409, unknown issues)
            if HTTP_403_REGEX not in line and HTTP_401_REGEX not in line and HTTP_409_REGEX not in line and HTTP_3XX_REGEX not in line and BLC_UNKNOWN_REGEX not in line and HTTP_UNDEFINED not in line and FALSE_SCALA_API_DOC_LINK not in line and FALSE_SCALA_API_DEPRECATED_LINK not in line and FALSE_PAPER_LINK not in line:
                current_page_broken = True
                current_page_broken_links += line.split(BROKEN_PAGE_START_REGEX)[1] + "\n"

        if line.startswith(PAGE_TEST_END_REGEX):
            if current_page_broken:
                broken_links_count += 1
                broken_links_summary += "\nURL - " + current_page
                broken_links_summary += "\nBroken Links\n" + current_page_broken_links

    return broken_links_count, broken_links_summary

# Command to check broken links
# Reference - https://www.npmjs.com/package/broken-link-checker
cmd = "blc https://mxnet.incubator.apache.org -ro"
broken_links_count = 0
broken_links_summary = ""

text_file = open("./blc_output.txt", 'w')
command_output = ""
print("Starting broken link test with command $ " + cmd)
try:
    command_output = check_output(cmd, stderr=STDOUT, shell=True)
except CalledProcessError as ex:
    if ex.returncode > 1:
        print("Failed to do broken link test. Console output : \n" + ex.output)
        sys.exit(ex.returncode)
    command_output = ex.output

text_file.write(command_output)
text_file.close()
broken_links_count, broken_links_summary = prepare_link_test_result(command_output)
# These START and END string in output is used to parse the script output in automated scripts and nightly jobs.
print("START - Broken links count")
print(broken_links_count)
print("END - Broken links count")

print("START - Broken links summary")

if broken_links_count == 0:
    print("No broken links in https://mxnet.incubator.apache.org")
    print("END - Broken links summary")
else:
    print(broken_links_summary)
    print("END - Broken links summary")
    # Fail the job as we found the broken links
    sys.exit(-1)
