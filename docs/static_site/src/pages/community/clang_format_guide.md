---
layout: page
title: Clang format
subtitle: Clang format in MXNet codebase for reviewers and contributors.
action: Contribute
action_url: /community/index
permalink: /community/clang_format_guide
---
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

Clang-format Guide and Tips
===================

This wiki page describes how to set up clang-format tool as a part of your worklow. Running the command given in the description will fix clang-format problem.


- Add `tools/lint/git-clang-format-13 ` to your `$PATH`. Once its added to your `$PATH`, running `git clang-format` will invoke it.
```bash
git clang-format
```


- To reformat chosen file just do: 
```bash
# `_FILE_NAME_` is the name of a file to be formatted.
# i - apply edits to files instead of displaying a diff
clang-format -i _FILE_NAME_
```

- To reformat all the lines in the latest git commit, just do: 
```bash
git diff -U0 --no-color HEAD^ | clang-format-diff.py -i -p1

```

- If you want to apply clang-format only to the changed lines in each commit do the following:
```bash
# If it's a child of origin/master, the following command-line could be used:
# If you want to run this command on another brnach, then origin/master needs to be replaced.
export COMMIT_SHA=$(git rev-list --ancestry-path origin/master..HEAD | tail -n 1)

git filter-branch --tree-filter 'git-clang-format $COMMIT_SHA^' -- $COMMIT_SHA..HEAD
```