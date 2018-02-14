# Contribute to MXNet

MXNet has been developed and is used by a group of active community members.
Please contribute to improve the project.
After your patch has been merged, remember to add your name to [CONTRIBUTORS.md](https://github.com/apache/incubator-mxnet/blob/master/CONTRIBUTORS.md).

## Code Contribution

### Core Library

- Follow the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html) for C++ code.
- Use doxygen to document all of the interface code.
- Use [RAII](http://en.cppreference.com/w/cpp/language/raii) to manage resources, including smart
 pointers like shared_ptr and unique_ptr as well as allocating in constructors and deallocating in
 destructors. Avoid explicit calls to new and delete when possible. Use make_shared and make_unique
  instead.
- To reproduce the linter checks, type ```make lint```. (You need to pip install pylint and cpplint
 before)

### Python Package

- Always add docstring to the new functions in numpydoc format.
- To reproduce the linter checks, type ```make lint```.

### R Package

#### Code Style
- Most of the C++ code in the R package relies heavily on [Rcpp](https://github.com/RcppCore/Rcpp).
- We follow the Google C++ Style Guide for C++ code. This allows us to maintain consistency with the rest of the project. It also allows us to check style automatically with a linter.
- To check the code style, type the following command at the root folder:
```bash
make rcpplint
```
- If necessary, disable the linter warning on certain lines with ```// NOLINT(*)``` comments.

#### Auto-Generated API
- Many MXNet APIs are exposed dynamically from Rcpp.
- mxnet_generated.R is the auto-generated API and documents for these functions.
- Remake the file by typing the following command at root folder:
```bash
make rcppexport
```
- Use this command only when there is an update to dynamic functions.

#### API Document
The document is generated using roxygen2. To remake the documents in the root folder, use the following command:
```bash
make roxygen.
```

#### R Markdown Vignettes
R Markdown vignettes are located on GitHub in [R-package/vignettes](https://github.com/apache/incubator-mxnet/tree/master/R-package/vignettes).
These R Markdown files aren't compiled. We host the compiled version on [doc/R-package](https://github.com/apache/incubator-mxnet/tree/master/R-package/).

To add a new R Markdown vignettes:

* Add the original R Markdown file to ```R-package/vignettes```
* Modify ```doc/R-package/Makefile``` to add the Markdown files to be built.
* Clone the [dmlc/web-data](https://github.com/dmlc/web-data) repo to  the  ```doc``` folder.
* Type the following command for the ```doc/R-package``` package:
```bash
make the-markdown-to-make.md
```
* This generates the markdown and the figures and places them into ```doc/web-data/mxnet/knitr```.
* Modify the ```doc/R-package/index.md``` to point to the generated markdown.
* Add the generated figure to the ```dmlc/web-data``` repo.
	* If you have already cloned the repo to doc, use ```git add```.
* Create a pull request for both the markdown  and ```dmlc/web-data```.
* You can also build the document locally with the following command: ```doc```
```bash
make html
```

### Test Cases

* All of our tests can be found in the GitHub repo in [this directory](https://github.com/apache/incubator-mxnet/tree/master/tests).
* We use Python nose for python test cases, and gtest for C++ unit tests.

### Examples

* Use cases and examples are on GitHub in [examples](https://github.com/apache/incubator-mxnet/tree/master/example)
* If you write a blog post or tutorial about or using MXNet, please tell us by creating an issue
in our github repo. We regularly feature high-quality contributed content from the community.

## Standard for Contributing APIs

Make sure to add documentation with any code you contribute. Follow these standards:

### API Documentation
* Document are created with Sphinx and [recommonmark](http://recommonmark.readthedocs.org/en/latest/).
* Follow [numpy doc standards](https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt#docstring-standard) and
some changes we made [MXNet doc standards](https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt#docstring-standard).
* If an API is implemented in Python or has a wrapper defined, the documentation and the examples reside
where the function is defined in `.py` file in [python/mxnet](https://github.com/apache/incubator-mxnet/tree/master/python/mxnet) folder. Same goes for other languages.
* If the API is dynamically generated from the MXNet backend, the documentation is in the C++ code(.cc
file) where the operator is registered in describe method of the `NNVM_REGISTER_OP`. The file and line
number for the function is usually printed with the API documentation on mxnet.io.
* A clear and concise description of the function and its behavior.
* List and describe each parameter with the valid input values, whether it is required or optional,
and the default value if the parameter is optional.
* Add an example to help the user understand the API better. If the example can be language-neutral
or is conceptual, add it in the C++ documentation. Make sure your example works
by running a Python version of the example.
  * If a concrete and simple language-specific example can further clarify the API and the API arguments, add the
example in language-specific files.
* Refer to these examples for guidance:- [Embedding](http://mxnet.io/api/python/ndarray/ndarray.html#mxnet.ndarray.Embedding) , [ROIPooling](http://mxnet.io/api/python/ndarray/ndarray.html#mxnet.ndarray.ROIPooling) , [Reshape](http://mxnet.io/api/python/ndarray/ndarray.html#mxnet.ndarray.Reshape).

### Testing and Rendering
* Make sure not to break any coding standards. Run
```bash
make lint
```
* You can build documents locally to proof them.

## Guidelines to submit a Pull Request
* Before submitting your contribution, rebase your code on the most recent version of the master:

```bash
    git remote add upstream https://github.com/apache/incubator-mxnet
    git fetch upstream
    git rebase upstream/master
```
* If you have multiple small commits,
   merge them into meaningful groups (use ```git rebase``` then ```squash```).
* Send the pull request.
* Fix problems reported by automatic checks.
* If you are contributing a new module, consider adding a test case in [tests](https://github.com/apache/incubator-mxnet/tree/master/tests).

### Resolving a Conflict with the Master

* Rebase to the current master:

 ```bash
    # The first two steps can be skipped after you do it once.
    git remote add upstream https://github.com/apache/incubator-mxnet
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
