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

import os
import argparse
from bs4 import BeautifulSoup as bs

parser = argparse.ArgumentParser(description="Add download package link.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--file_path', type=str, default='docs/_build/html/install/index.html',
                        help='file to be modified')
parser.add_argument('--current_version', type=str, default='master',
                        help='Current version')

if __name__ == '__main__':
    args = parser.parse_args()
    tag = args.current_version

    src_url = "https://www.apache.org/dyn/closer.cgi/incubator/" \
              "mxnet/%s-incubating/apache-mxnet-src-%s-incubating.tar.gz" % (tag, tag)
    pgp_url = "https://www.apache.org/dyn/closer.cgi/incubator/" \
              "mxnet/%s-incubating/apache-mxnet-src-%s-incubating.tar.gz.asc" % (tag, tag)
    sha_url = "https://www.apache.org/dyn/closer.cgi/incubator/" \
              "mxnet/%s-incubating/apache-mxnet-src-%s-incubating.tar.gz.sha" % (tag, tag)
    md5_url = "https://www.apache.org/dyn/closer.cgi/incubator/" \
              "mxnet/%s-incubating/apache-mxnet-src-%s-incubating.tar.gz.md5" % (tag, tag)

    download_str = "<div class='btn-group' role='group'>"
    download_str += "<div class='download_btn'><a href=%s>" \
                    "<span class='glyphicon glyphicon-download-alt'></span>" \
                    " Source for %s</a></div>" % (src_url, tag)
    download_str += "<div class='download_btn'><a href=%s>PGP</a></div>" % (pgp_url)
    download_str += "<div class='download_btn'><a href=%s>SHA-256</a></div>" % (sha_url)
    download_str += "<div class='download_btn'><a href=%s>MD5</a></div>" % (md5_url)
    download_str += "</div>"

    with open(args.file_path, 'r') as html_file:
        content = bs(html_file, 'html.parser')
    download_div = content.find(id="download-source-package")
    download_div['style'] = "display:block"
    download_div.append(download_str)
    outstr = str(content).replace('&lt;', '<').replace('&gt;', '>')
    with open(args.file_path, 'w') as outf:
        outf.write(outstr)