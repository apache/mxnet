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
#!/bin/bash
wget https://s3.us-east-2.amazonaws.com/scala-infer-models/resnet-18/resnet-18-symbol.json -P target/resnet18/ -q
wget https://s3.us-east-2.amazonaws.com/scala-infer-models/resnet-18/resnet-18-0000.params -P target/resnet18/ -q
wget https://s3.us-east-2.amazonaws.com/scala-infer-models/resnet-18/synset.txt -P target/resnet18/ -q
wget https://s3.amazonaws.com/model-server/inputs/kitten.jpg -P target/ -q
mkdir target/images
for i in {1..11}; do cp target/kitten.jpg "target/images/kitten$i.jpg"; done