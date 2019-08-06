# R binding docs

Follow these steps to build the MXNet R website from within the Rsite/ parent directory

(homepage will appear in: Rsite/build/index.html)

make clean

make


## Setup

You need to have necessary software installed, and Jupyter Notebook with R kernel and notedown compatibility (to compile tutorials).

1) Install MXNet R version:
http://mxnet.incubator.apache.org/install/index.html?platform=Linux&language=R&processor=CPU

2) Install other necessary R packages:
Start R in your terminal (Note: cannot use R studio here to ensure Jupyter kernel can access these packages as well):

install.packages(c('mxnet', 'repr', 'IRdisplay', 'evaluate', 'crayon', 'pbdZMQ', 'devtools', 'uuid', 'digest'))

3) Install Jupyter Notebook with R kernel and notedown compatibility:

https://www.datacamp.com/community/blog/jupyter-notebook-r

https://github.com/aaren/notedown

Also add the following line to your Juypter config file (`~/.jupyter/jupyter_notebook_config.py`):

c.NotebookApp.contents_manager_class = 'notedown.NotedownContentsManager'

So that you can then simply run Jupyter notebook and edit .md files just as if they are standard notebook .ipynb files.

4) Install the necessary mxtheme python package (used for website style):

pip install https://github.com/mli/mx-theme/tarball/master

5) Install sphinx & pandoc (used for processing Notebooks with Sphinx):

http://pandoc.org/installing.html

6) You will also need to have the .Rd documentation files already generated for the 'mxnet' R package.
Up-to-date R documentation files can be obtained by building MXNet from the source:

https://github.com/apache/incubator-mxnet

These files will then be located in the directory: PATH:/mxnet/R-package/man/

Copy all of them into subdirectory named: source/api/man/   (replacing existing files as needed)

You can check out: scripts/Rdoc2SphinxDoc.R to see locations of all files that the sphinx-documentation-generation script depends on.

7) For the tutorials, in addition to the .md Notebook files, you must have the following files available in the same directory:

data/train.csv, data/test.csv (for DigitsClassification tutorial)

Inception/Inception_BN-symbol.json (for ClassifyImageWithPretrainedModel tutorial)

You should also ensure the file tutorial/index.rst lists all the tutorials you wish to include on the website (and in the proper order).

Note: If creating future tutorials will depend on new object files, you must update the makefile to ensure these new object files get copied into the "mxnetRtutorials/" subdirectory which is subsequently zipped and made available for user-download (so they can run the Jupyter notebooks themselves).  


## Detailed descriptions of Makefile operations:

### build documentation:

cd ./source/

Rscript ./scripts/Rdoc2SphinxDoc.R

### build tutorials:

cd ./source/tutorial/

bash scripts/convertNotebooks.sh

### build sphinx page:

cd ./ # (must be in home directory containing source/ and build/ subdirectories)

sphinx-build -b html source/ build/


## TODOs:

In ClassifyImageWithPretrainedModel.md tutorial:
    model-zoo link should be updated to new MXNet site ("Model Zoo"); as should link for downloading pre-trained network ("this link": http://data.mxnet.io/mxnet/data/Inception.zip). 

In Symbol.md tutorial:
    Link (Symbolic Configuration and Execution in Pictures) should be updated to point at new MXNet website.

In index.rst R homepage:  

- R-MXnet installation instructions missing. 

- toctree table of contents should be updated to match Python page side-bar on: https://beta.mxnet.io/

Disqus does not work.

Makefile process should be streamlined.

None of the R tutorial notebook output is currently being rendered in Sphinx.

The link to download all Notebooks (on this page: http://beta.mxnet.io/r/tutorial/index.html) points to. empty zip directory.


## Remaining Problems in underlying Rd documentation:

1) Biggest issue is many of the code examples (or “usage”) of certain functions are actually in Python, not R.
See for example: mx.symbol.ones_like, mx.symbol.square, etc. There’s generally Python code appearing all over the place in the R documentation, which should probably be translated to R.

2) mx.io.MNISTIter: “, optional” in description of prefetch.buffer argument (extra comma should be removed)

3) mx.nd.log1p: bulleted list has nonuniform indentation levels

4) mx.nd.relu: Description "Computes rectified linear" is missing "activation"

5) mx.nd.cast.storage.Rd \details{} block begins too early.  This is actually a problem in many of the .Rd files: the \description{} and \details{} are actually contiguous blocks of text which have been arbitrarily split into these separate fields.  Where the split between \description{} and \details{} takes place should be more sensibly chosen (definitely not in the middle of a sentence or mathematical definition).  See also: mx.nd.dot, mx.nd.abs for other examples of this issue, which occurs all over the place.

6) mx.nd.Embedding: contains link to python API.  Should probably specify this as “the Optimization API for the Python version of MXNet”

7) mx.opt.sgd: “momentumvalue” should be “momentum value”

8) predict.MXFeedForwardModel: description of argument “array.batch.size” is missing period or linebreak after “mx.gpu(i)”.

9) mx.ctx.default: argument name "new, " should be “new” (without comma)

10) rnn.graph.unroll: \description{} field starts in lower-case, while all other functions descriptions start upper-case.

11) mx.nd.sgd.update: description "SDG" spelled wrong.

12) There is both: mx.nd.ctc.loss and mx.symbol.ctc_loss (seems redundant and one should probably be removed from the R package)

13) Function names ending in v1 are deprecated, e.g. “mx.nd.Convolution.v1” (should probably be removed from the R package)

14) mx.symbol.pad is missing .Rd documentation (does not automatically link to mx.symbol.Pad documention in R), whereas mx.nd.pad properly links to mx.nd.Pad documentation.

15) mx.symbol.Pad & mx.nd.Pad: "\title{pad:Pads" is missing space between 'pad:' and the subsequent description.  More generally, should probably remove the function name from \title{} blocks to standardize things, since the function name is sometimes listed there (followed by colon) and sometimes not.  However, please do not start including the function name in the \title{} field without colon, since my parser to convert the documentation into a Sphinx website currently critically relies on this property to hide the redundancy of the function name appearing again in the \title{} field.
