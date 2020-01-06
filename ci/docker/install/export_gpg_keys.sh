#!/usr/bin/env bash

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

gpg --keyserver keyserver.ubuntu.com --recv 2EE0EA64E40A89B84B2DF73499E82A75642AC823
gpg --output sbt.gpg --export scalasbt@gmail.com
gpg --keyserver keyserver.ubuntu.com --recv E084DAB9
gpg --output r.gpg --export marutter@gmail.com
