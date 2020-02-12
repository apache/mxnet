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

all:
	julia --color=yes --project=./ -e \
	  'using Pkg; \
	   Pkg.develop(PackageSpec(name="MXNet", path = joinpath(pwd(), "..")))'
	julia --color=yes --project=./ ./make.jl
	pip install --user Markdown==3.1 \
          mkdocs==1.0.4 \
	  mkdocs-material==4.6.0 \
	  pygments==2.5.2 \
	  pymdown-extensions==6.2.1 \
	  python-markdown-math==0.6
	export LC_ALL="C.UTF-8"
	export LANG="C.UTF-8"
	echo $(LC_ALL)
	echo $(LANG)
	LC_ALL="C.UTF-8" ~/.local/bin/mkdocs build

clean:
	rm -rvf venv
	rm -rvf build
	rm -rvf site
