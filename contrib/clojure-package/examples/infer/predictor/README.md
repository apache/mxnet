<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# predictor

Run model prediction using clojure infer package.

## Installation

Before you run this example, make sure that you have the clojure package installed.
In the main clojure package directory, do `lein install`. Then you can run
`lein install` in this directory.

## Usage

```
$ chmod +x scripts/get_resnet_18_data.sh
$ ./scripts/get_resnet_18_data.sh
$
$ lein run -- --help
$ lein run -- -m models/resnet-18/resnet-18 -i images/kitten.jpg
$
$ lein uberjar
$ java -jar target/predictor-0.1.0-SNAPSHOT-standalone.jar --help
$ java -jar target/predictor-0.1.0-SNAPSHOT-standalone.jar \
    -m models/resnet-18/resnet-18 -i images/kitten.jpg
```
