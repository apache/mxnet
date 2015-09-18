Contribute to MXNet
===================
MXNet has been developed and used by a group of active community members.
Everyone is more than welcome to contribute. It is a way to make the project better and more accessible to more users.
* Please add your name to [CONTRIBUTORS.md](../CONTRIBUTORS.md) after your patch has been merged.

Code Style
----------
- Follow Google C style for C++.
- We use doxygen to document all the interface code.
- We use numpydoc to document all the python codes.
- You can reproduce the linter checks by typing ```make lint```

Contribute to Documents
-----------------------
* The document is created using sphinx and [recommonmark](http://recommonmark.readthedocs.org/en/latest/)
* You can build document locally to see the effect.

Contribute to Testcases
-----------------------
* All the testcases are in [tests](../tests)
* We use python nose for python test cases and gtest for c++ unittests.

Contribute to Examples
----------------------
* Usecases and examples will be in [example](../example)
* We are super excited to hear about your story, if you have blogposts,
  tutorials code solutions using mxnet, please tell us and we will add
  a link in the example pages.

Submit a Pull Request
---------------------
* Before submit, please rebase your code on the most recent version of master, you can do it by
```bash
git remote add upstream https://github.com/dmlc/mxnet
git fetch upstream
git rebase upstream/master
```
* If you have multiple small commits,
  it might be good to merge them together(use git rebase then squash) into more meaningful groups.
* Send the pull request!
  - Fix the problems reported by automatic checks
  - If you are contributing a new module, consider add a testcase in [tests](../tests)
