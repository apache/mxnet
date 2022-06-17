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

all: html

html:
	mkdir -p build
	wget -O src/assets/js/jquery-3.3.1.min.js https://code.jquery.com/jquery-3.3.1.min.js
	wget -O src/assets/img/mxnet-icon.png https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/image/mxnet-icon.png
	wget -O src/assets/docsearch.min.css 'https://cdn.jsdelivr.net/npm/docsearch.js@2/dist/cdn/docsearch.min.css'
	wget -O src/assets/js/docsearch.min.js 'https://cdn.jsdelivr.net/npm/docsearch.js@2/dist/cdn/docsearch.min.js'
	wget -O src/assets/js/fontawesome.js 'https://use.fontawesome.com/releases/v5.0.12/js/all.js'
	wget -O src/assets/js/buttons.js 'https://buttons.github.io/buttons.js'
	wget -O src/assets/js/platform.js 'https://apis.google.com/js/platform.js'
	cd src && bundle install && JEKYLL_ENV=production bundle exec jekyll build --config _config_prod.yml -d ../build/html && cd ..
	wget https://mxnet-website-static-artifacts.s3.us-east-2.amazonaws.com/versions.zip && unzip versions.zip -d build/html
	find build/html/ -type d -name '__MACOSX' -exec rm -rf {} +
	find build/html/ -type f -name '.DS_Store' -exec rm -rf {} +
	rm versions.zip

clean:
	rm -rf build
