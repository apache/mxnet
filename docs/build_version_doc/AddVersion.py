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

parser = argparse.ArgumentParser(description="Manipulate index page",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--file_path', type=str, default='mxnet/docs/_build/html/',
                        help='file to be modified')
parser.add_argument('--current_version', type=str, default='master',
                        help='Current version')
parser.add_argument('--root_url', type=str, default='https://mxnet.incubator.apache.org/',
                        help='Root URL')
parser.add_argument('--tag_default', type=str, default='master', help='Default Tag')

if __name__ == '__main__':
    args = parser.parse_args()

    root_url = args.root_url
    tag_list = list()
    with open('tag_list.txt', 'r') as tag_file:
        for line in tag_file:
            tag_list.append(line.lstrip().rstrip())

    version_str = '<span id="dropdown-menu-position-anchor-version" ' \
                  'style="position: relative">' \
                  '<a href="#" class="main-nav-link dropdown-toggle" data-toggle="dropdown" ' \
                  'role="button" aria-haspopup="true" aria-expanded="true">Versions(%s)<span class="caret">' \
                  '</span></a><ul id="package-dropdown-menu" class="dropdown-menu">' % (args.current_version)
    version_str_mobile = '<li id="dropdown-menu-position-anchor-version-mobile" class="dropdown-submenu" ' \
                         'style="position: relative">' \
                         '<a href="#" tabindex="-1">Versions(%s)</a><ul class="dropdown-menu">' % (args.current_version)
    for i, tag in enumerate(tag_list):
        url = root_url + 'versions/%s/index.html' % (tag)
        version_str += '<li><a class="main-nav-link" href=%s>%s</a></li>' % (url, tag)
        version_str_mobile += '<li><a tabindex="-1" href=%s>%s</a></li>' % (url, tag)
    version_str += '</ul></span>'
    version_str_mobile += '</ul></li>'

    for path, subdirs, files in os.walk(args.file_path):
        for name in files:
            if not name.endswith('.html'):
                continue
            if 'install' in path:
                print("Skipping this path: {}".format(path))
                continue
            with open(os.path.join(path, name), 'r') as html_file:
                content = bs(html_file, 'html.parser')
            navbar = content.find(id="main-nav")
            navbar_mobile = content.find(id="burgerMenu")
            outstr = str(content)
            if navbar and navbar_mobile:
                version_tag = content.find(id="dropdown-menu-position-anchor-version")
                version_tag_mobile = content.find(id="dropdown-menu-position-anchor-version-mobile")
                if version_tag:
                    version_tag.extract()
                if version_tag_mobile:
                    version_tag_mobile.extract()
                navbar.append(version_str)
                navbar_mobile.append(version_str_mobile)
                # The following causes rendering errors in code blocks; refer to #12168 and #12524
                outstr = str(content).replace('&lt;', '<').replace('&gt;', '>')
            # Fix link
            if args.current_version == tag_list[0]:
                print("Fixing " + os.path.join(path, name))
                outstr = outstr.replace('https://mxnet.io', 'https://mxnet.incubator.apache.org')
                outstr = outstr.replace('http://mxnet.io', 'https://mxnet.incubator.apache.org')
            else:
                outstr = outstr.replace('https://mxnet.io', 'https://mxnet.incubator.apache.org/'
                                                                'versions/%s' % (args.current_version))
                outstr = outstr.replace('http://mxnet.io', 'https://mxnet.incubator.apache.org/'
                                                               'versions/%s' % (args.current_version))

            with open(os.path.join(path, name), "w") as outf:
                outf.write(outstr)
