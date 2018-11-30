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

SCALA_VERSION_PROFILE := 2.11
SCALA_VERSION := 2.11.8
MXNET_VERSION := "[1.3.0-SNAPSHOT,)"

MXNET_REPO = https://repository.apache.org/content/repositories/snapshots

ifeq ($(OS),Windows_NT)
	UNAME_S := Windows
else
	UNAME_S := $(shell uname -s)
endif

ifeq ($(UNAME_S), Windows)
  # TODO: currently scala package does not support windows
	SCALA_PKG_PROFILE := windows
else
	ifeq ($(UNAME_S), Darwin)
		SCALA_PKG_PROFILE := osx-x86_64-cpu
	else
		SCALA_PKG_PROFILE := linux-x86_64
		ifeq ($(USE_CUDA), 1)
          SCALA_PKG_PROFILE := $(SCALA_PKG_PROFILE)-gpu
        else
          SCALA_PKG_PROFILE := $(SCALA_PKG_PROFILE)-cpu
        endif
	endif
endif

PROFILES := -Ptest
ifeq ($(UNIT), 1)
  PROFILES := "$(PROFILES),unittest"
endif
ifeq ($(INTEGRATION), 1)
  PROFILES := "$(PROFILES),integrationtest"
endif

ifneq ($(UNIT), 1)
	ifneq ($(INTEGRATION), 1)
		PROFILES := "$(PROFILES),unittest,integrationtest"
	endif
endif


clean:
	(mvn clean -Dmxnet.profile=$(SCALA_PKG_PROFILE) \
		-Dmxnet.scalaprofile=$(SCALA_VERSION_PROFILE) \
		-Dmxnet.version=$(MXNET_VERSION) \
		-Dscala.version=$(SCALA_VERSION))

testinstall:
	(mvn integration-test -Dmxnet.profile=$(SCALA_PKG_PROFILE) \
    $(PROFILES) \
		-Dmxnet.scalaprofile=$(SCALA_VERSION_PROFILE) \
		-Dmxnet.version=$(MXNET_VERSION) \
		-Dscala.version=$(SCALA_VERSION))

testlocal:
	(mvn integration-test -Dmxnet.profile=$(SCALA_PKG_PROFILE) \
    $(PROFILES),fromLocal \
		-Dmxnet.scalaprofile=$(SCALA_VERSION_PROFILE) \
		-Dmxnet.version=$(MXNET_VERSION) \
		-Dscala.version=$(SCALA_VERSION))

testsnapshot:
	(mvn integration-test -Dmxnet.profile=$(SCALA_PKG_PROFILE) \
    $(PROFILES),fromSnapshots \
		-Dmxnet.scalaprofile=$(SCALA_VERSION_PROFILE) \
		-Dmxnet.repo=$(MXNET_REPO) \
		-Dmxnet.version=$(MXNET_VERSION) \
		-Dscala.version=$(SCALA_VERSION))
