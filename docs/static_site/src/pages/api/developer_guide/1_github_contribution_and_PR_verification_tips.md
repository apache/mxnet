---
layout: page_category
title:  GitHub contribution and PR verification tips 
category: Developer Guide
permalink: /api/dev-guide/github_contribution_and_PR_verification_tips
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

# GitHub contribution and PR verification tips 

Use this page for general git workflow tips. 

## Setup and configure

It is recommended that you fork the MXNet repo, and then set the original repo as an upstream remote repo. 

Fork [https://github.com/apache/mxnet](https://github.com/apache/mxnet) then:

```
git clone --recursive https://github.com/your_username/mxnet
cd mxnet
git remote add upstream https://github.com/apache/mxnet
```

Once `upstream` was added, then create a branch for your contribution.


```
git branch your-contribution-branch
```

Note that you can incorporate the changes from `upstream` to any of your local branches during or after development via: 

```
git fetch upstream
git rebase upstream/master
```

See [this stackoverflow discussion](https://stackoverflow.com/questions/3357122/git-pull-vs-git-fetch-vs-git-rebase) for more details about difference between `git pull`, `git rebase` and `git merge`.

Since Apache MXNet 3rd party git submodules, to update their changes on your branch after rebase, you can run:

```
git submodule update --recursive
```

## Save your local changes for future

During development, you can save your current changes in your branch before committing anything. For example to go to another branch to do something else via:


```
git stash save
```

To restore the changes so that they can be added to a commit use:


```
git stash pop
```


To drop the changes, use:

```
git stash drop
```

## Reset

Sometimes, if you want to wipe out the changes you have made you can use:

```
git reset --hard
```

Be very careful since hard-reset removes any of the changes and you’ll be back to the HEAD commit. To remove all the changed before a commit given its commit-SHA you can use `git reset --hard commit-SHA` or `git reset --hard HEAD~2` to remove relative to the first two commits on top of HEAD.

However, sometimes it’s useful to keep the files/changes staged when moving the HEAD which can be done via 
`git reset --soft`. All of the files changed between the original HEAD and the commit will be staged.

In [summary](https://stackoverflow.com/a/50022436),


* **`--soft`**: **uncommit** changes, changes are left staged (*index*).
* **`--mixed`** *(default)*: **uncommit + unstage** changes, changes are left in *working tree*.
* **`--hard`**: **uncommit + unstage + delete** changes, nothing left.



## Recover a previous commit after reset

Sometimes you might mistakenly reset a branch to a wrong commit. When that happens, you can use the following command to show the list of recent commits:


```
git reflog
```

Once you get the right hashtag, you can use git reset again to change the head to the right commit.


## How to resolve conflict with master

Sometimes when rebasing to the most recent master as explained above, git may show you there are some conflicts which it cannot resolve. These changes will not be merged. For examples, your file `conflict.py` has some conflicts with the master branch. Here you need to:

* manually modify the file to resolve the conflict.
* After you resolved the conflict, mark it as resolved by:

```
git add conflict.py
```

* Then you can continue rebase by:

```
git rebase --continue
```

* Finally push to your fork, you may need to **force push** here:

```
git push --force
```

**Note** that force push is okay when it’s on your branch and you are the only one who is using that branch. Otherwise, it can have bad consequences as it’s rewritten the history.


## How to group multiple commits into one

Sometimes, you may have added a lot of related commits suitable to be grouped/combined together to create one meaningful atomic commit. For example, when later commits are only fixes to previous ones, in your PR. 
If you haven’t configured your default git editor, do the following once:

```
git config core.editor the-editor-you-like
```

Assume we want to merge the last 3 commits.

```
git rebase -i HEAD~3
```

1. It will pop up an text editor. Set the **first commit as pick,** and **change later ones to squash**.
2. After you saved the file, it will pop up another text editor to ask you modify the combined commit message.
3. Push the changes to your fork, you need to force push.

```
git push --force
```

**Note** that force push is okay when it’s on your branch and you are the only one who is using that branch. Otherwise, it can have bad consequences as it’s rewritten the history.


## Apply only k-latest commits on to the master

Sometimes it is useful to only apply your k-latest changes on top of the master. This usually happens when you have other m-commits that are already merged before these k-commits. Directly rebase against the master might cause merge conflicts on these first m-commits (which can be safely discarded).

You can instead use the following command:


```
# k is the concrete number. Put HEAD~2 for the last 1 commit.
git rebase --onto upstream/master HEAD~k
```

You can then force push to the master `git push --force`. Note that the above command will discard all the commits before the last k ones.


## What is the consequence of force push

The last three tips require the force push, this is because we altered the path of the commits. **It is fine to force push to your own fork, as long as the commits changed are only yours.** In case there are multiple collaborators who use your branch there is a safer option `git push --force-with-lease.`


## PR verification

When sending a pull request, remember to add some tests. During the development, one can set `MXNET_TEST_COUNT=1000/10000` to test on some randomly selected test cases. This makes the testing and development cycle faster. Moreover, some test results might change due to the seed in pseudo-random number generator. To fix the seed during testing, set `MXNET_TEST_SEED=your seed number`.
