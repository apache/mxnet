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

# MXNet.io v2

1. Install Jekyll https://jekyllrb.com/docs/installation/

This is for hosting the mxnet.io beta website

serve for test:
```
cd src && JEKYLL_ENV=development bundle exec jekyll serve
```

build for beta github pages:
```
cd src && JEKYLL_ENV=production bundle exec jekyll build --config _config_beta.yml -d ../docs && cd ..
```


build for release:
```
cd src && JEKYLL_ENV=production bundle exec jekyll build --config _config_prod.yml -d ../release && cd ..
```

test:

https://thomasdelteil.github.io/mxnet.io-v2/
