---
layout: page
title: Write Document and Tutorials
subtitle: Guidelines on documentation to help the community.
action: Contribute
action_url: /community/index
permalink: /community/document
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

Write Document and Tutorials
============================

We use the [Sphinx](http://sphinx-doc.org) for the main documentation.
Sphinx support both the reStructuredText and markdown. When possible, we
encourage to use reStructuredText as it has richer features. Note that
the python doc-string and tutorials allow you to embed reStructuredText
syntax.

Document Python
---------------

We use [numpydoc](https://numpydoc.readthedocs.io/en/latest/) format to
document the function and classes. The following snippet gives an
example docstring. We always document all the public functions, when
necessary, provide an usage example of the features we support(as shown
below).

```python
def myfunction(arg1, arg2, arg3=3):
    """Briefly describe my function.

    Parameters
    ----------
    arg1 : Type1
        Description of arg1

    arg2 : Type2
        Description of arg2

    arg3 : Type3, optional
        Description of arg3

    Returns
    -------
    rv1 : RType1
        Description of return type one

    Examples
    --------
    .. code:: python

        # Example usage of myfunction
        x = myfunction(1, 2)
    """
    return rv1
```

Be careful to leave blank lines between sections of your documents. In
the above case, there has to be a blank line before
[Parameters]{.title-ref}, [Returns]{.title-ref} and
[Examples]{.title-ref} in order for the doc to be built correctly. To
add a new function to the doc, we need to add the
[sphinx.autodoc](http://www.sphinx-doc.org/en/master/ext/autodoc.html)
rules to the
[/docs/python_docs/python](https://github.com/apache/incubator-mxnet/tree/master/docs/python_docs/python)).
You can refer to the existing files under this folder on how to add the
functions.

Document C++
------------

We use the doxgen format to document c++ functions. The following
snippet shows an example of c++ docstring.

```cpp
/*!
 * \brief Description of my function
 * \param arg1 Description of arg1
 * \param arg2 Description of arg2
 * \returns describe return value
 */
int myfunction(int arg1, int arg2) {
  // When necessary, also add comment to clarify internal logic
}
```

Besides documenting function usages, we also highly recommend
contributors to add comments about code logic to improve readability.

Write Tutorials
---------------

We use the [notedown](https://github.com/aaren/notedown) to write Jupyter notebooks
in Markdown as Python tutorials. You can find the source code under
[/docs/python_docs/python/tutorials](https://github.com/apache/incubator-mxnet/tree/master/docs/python_docs/python/tutorials).


The tutorial code will run on our build server to generate the document
page and the tutorial page will show the result of executing the Jupyter notebook.


Application Examples
--------------------

Our deep learning examples are maintained in [apache/incubator-mxnet-examples](http://github.com/apache/incubator-mxnet-examples)
and are checked regularly by CI to ensure quality.

<script async defer src="https://buttons.github.io/buttons.js"></script>
<script src="https://apis.google.com/js/platform.js"></script>
