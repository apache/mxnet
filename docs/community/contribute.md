# Contribute to MXNet

MXNet has been developed and is used by a group of active community members.
Please contribute to improve the project.
After your patch has been merged, remember to add your name to [CONTRIBUTORS.md](https://github.com/dmlc/mxnet/blob/master/CONTRIBUTORS.md).

## Guidelines

* [Submitting a Pull Request](#submitting-a-pull-request)
* [Resolving a Conflict with the Master](#resolving-a-conflict-with-the-master)
* [Combining Multiple Commits](#combining-multiple-commits)
* [What Is the Consequence of Forcing a Push?](#what-is-the-consequence-of-forcing-a-push)
* [Document](#documents)
* [Test Cases](#test-cases)
* [Examples](#examples)
* [Tutorials](#tutorials)
* [Core Library](#core-library)
* [Python Package](#python-package)
* [R Package](#r-package)

### Submitting a Pull Request
* Before submitting your contribution, rebase your code on the most recent version of the master:

```bash
    git remote add upstream https://github.com/dmlc/mxnet
    git fetch upstream
    git rebase upstream/master
```
* If you have multiple small commits,
   merge them into meaningful groups (use ```git rebase``` then ```squash```).
* Send the pull request.
* Fix problems reported by automatic checks.
* If you are contributing a new module, consider adding a test case in [tests](https://github.com/dmlc/mxnet/tree/master/tests).


### Resolving a Conflict with the Master

* Rebase to the current master:

 ```bash
    # The first two steps can be skipped after you do it once.
    git remote add upstream https://github.com/dmlc/mxnet
    git fetch upstream
    git rebase upstream/master
 ```

*  Git might show some conflicts that prevent merging, for example,  ```conflicted.py```.
	* Manually modify the file to resolve the conflict.
	* After you resolve the conflict, mark it as resolved by using:

```bash
git add conflicted.py.
```


* Continue rebasing by using this command:

 ```bash
    git rebase --continue
 ```

* Finally push to your fork. You might need to force the  push:

 ```bash
    git push --force
 ```

### Combining Multiple Commits
If you are submitting multiple commits with later commits that are just fixes to previous ones, you can combine commits into meaningful groups before creating a push request.


* Before doing so, configure Git's default editor if you haven't already done that:

```bash
git config core.editor the-editor-you-like
```
* Assuming that you want to merge last the last three commits, type the following commands:

```bash
git rebase -i HEAD~3
```

* In the text editor that appears, set the first commit as ```pick```, and change later ones to ```squash```.

* After you save the file, another text editor will appear and ask you to modify the combined commit message.

* Push the changes to your fork by forcing a push:

```bash
git push --force.
```

### What Is the Consequence of Forcing a Push?
The previous two tips require forcing a push because we altered the path of the commits.
It's fine to force a push to your own fork, as long as only your commits are changed.

## Documents

* Document are created with Sphinx and [recommonmark](http://recommonmark.readthedocs.org/en/latest/).
* You can build documents locally to proof them.

## Test Cases

* All of the test cases are in GitHub in [tests](https://github.com/dmlc/mxnet/tree/master/tests)
* We use python nose for python test cases, and gtest for c++ unit tests.

## Examples

* Use cases and examples are on GitHub in [examples](https://github.com/dmlc/mxnet/tree/master/example)
* If you have blog posts about MXNet or
  tutorials that use MXNet, please tell us and we'll add
  a link to them in the examples on GitHub.

## Tutorials

Want to contribute an MXNet tutorial? To get started, download the [tutorial template](https://github.com/dmlc/mxnet/tree/master/example/MXNetTutorialTemplate.ipynb).

## Core Library

- We follow the Google C++ Style Guide for C++ code.
- We use doxygen to document all of the interface code.
- To reproduce the linter checks, type ```make lint```.

## Python Package

- Always add docstring to the new functions in numpydoc format.
- You can reproduce the linter checks by typing ```make lint```.

## R Package

### Code Style
- Most of the C++ code in the R package relies heavily on [Rcpp](https://github.com/RcppCore/Rcpp).
- We follow the Google C++ Style Guide for C++ code. This allows us to maintain consistency with the rest of the project. It also allows us to check style automatically with a linter.
- To check the code style, type the following command at the root folder:
```bash
make rcpplint
```
- If necessary, you can disable the linter warning on certain lines with ```// NOLINT(*)``` comments.

### Auto-Generated API
- Many MXNet APIs are exposed dynamically from Rcpp.
- [mx_generated.R](R/mx_generated.R) is the auto-generated API and documents for these functions.
- Remake the file by typing the following command at root folder:
```bash
make rcppexport
```
- Use this command only when there is an update to dynamic functions.

### API Document
The document is generated using roxygen2. To remake the documents in the  root folder, use the following command:
```bash
make roxygen.
```

### R Markdown Vignettes
R Markdown vignettes are located on GitHub in [R-package/vignettes](https://github.com/dmlc/mxnet/tree/master/R-package/vignettes).
These R Markdown files aren't compiled. We host the compiled version on [doc/R-package](R-package).

To add a new R Markdown vignettes:

* Add the original R Markdown file to ```R-package/vignettes```
* Modify ```doc/R-package/Makefile``` to add the Markdown files to be built.
* Clone the [dmlc/web-data](https://github.com/dmlc/web-data) repo to  the  ```doc``` folder.
* Type the following command for the ```doc/R-package``` package:
```bash
make the-markdown-to-make.md
```
* This generates the markdown, and the figures and places them into ```doc/web-data/mxnet/knitr```.
* Modify the ```doc/R-package/index.md``` to point to the generated markdown.
* Add the generated figure to the ```dmlc/web-data``` repo.
	* If you have already cloned the repo to doc, use ```git add```.
* Create a pull request for both the markdown  and ```dmlc/web-data```.
* You can also build the document locally with the following command: ```doc```
```bash
make html
```
This prevents radically increasing the size of the repo with generated images sizes.
