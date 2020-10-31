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

# markdown files that will not be evaluated, simply copy to build/
PURE_MARKDOWN =  ./README.md
# markdown files that will be evaluated and then saved as ipynb files
IPYNB_MARKDOWN = $(shell find . -not -path "./build/*" -not -path "*.ipynb_checkpoints*" -name "*.md" | sort -h)
# RST files will be simply coped to build/
RST = $(shell find . -not -path "./build/*" -not -path "*.ipynb_checkpoints*" -name "*.rst")

OBJ = $(patsubst %.rst, build/%.rst, $(RST)) \
	$(patsubst %.md, build/%.ipynb, \
		$(filter-out $(PURE_MARKDOWN), $(IPYNB_MARKDOWN)))

build/%.ipynb: %.md
	@mkdir -p $(@D)
	python3 scripts/md2ipynb.py $< $@


build/%.rst: %.rst
	@mkdir -p $(@D)
	# python3 scripts/process_rst.py $< $@
	cp $< $@

build/%: %
	@mkdir -p $(@D)
	@cp -r $< $@


html: $(OBJ)
	mkdir -p build;
	cp Makefile_sphinx build/Makefile;
	cp -n -r ../_static build/ || true;
	sphinx-autogen build/api/*.rst build/api/**/*.rst   -t build/_templates/;
	# make -C build linkcheck doctest html
	make -C build html;
	sed -i.bak 's/33\,150\,243/23\,141\,201/g'  build/_build/html/_static/material-design-lite-1.3.0/material.blue-deep_orange.min.css;
	make update_github_link

update_github_link:
	for f in $(shell find build/_build/html/tutorials -type f -name '*.html'); do \
		echo "Updating github edit link for $$f."; \
		sed -i.bak 's/\(href="https:\/\/github.com\/apache\/mxnet\/edit\/master\/docs\/python_docs\/python\/tutorials\/[^"]*\).ipynb"/\1.md"/g' $$f; \
	done;

clean:
	rm -rf build
