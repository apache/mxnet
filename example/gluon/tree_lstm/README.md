
# Tree-Structured Long Short-Term Memory Networks
This is a [MXNet Gluon](https://mxnet.io/) implementation of Tree-LSTM as described in the paper [Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks](http://arxiv.org/abs/1503.00075) by Kai Sheng Tai, Richard Socher, and Christopher Manning. 

### Requirements
- Python (tested on **3.6.5**, should work on **>=2.7**)
- Java >= 8 (for Stanford CoreNLP utilities)
- Other dependencies are in `requirements.txt`
Note: Currently works with MXNet 1.3.0.

### Usage
Before delving into how to run the code, here is a quick overview of the contents:
 - Use the script `fetch_and_preprocess.sh` to download the [SICK dataset](http://alt.qcri.org/semeval2014/task1/index.php?id=data-and-tools), [Stanford Parser](http://nlp.stanford.edu/software/lex-parser.shtml) and [Stanford POS Tagger](http://nlp.stanford.edu/software/tagger.shtml), and [Glove word vectors](http://nlp.stanford.edu/projects/glove/) (Common Crawl 840) -- **Warning:** this is a 2GB download!), and additionally preprocess the data, i.e. generate dependency parses using [Stanford Neural Network Dependency Parser](http://nlp.stanford.edu/software/nndep.shtml).
- `main.py`does the actual heavy lifting of training the model and testing it on the SICK dataset. For a list of all command-line arguments, have a look at `python main.py -h`.
- The first run caches GLOVE embeddings for words in the SICK vocabulary. In later runs, only the cache is read in during later runs.

Next, these are the different ways to run the code here to train a TreeLSTM model.
#### Local Python Environment
If you have a working Python3 environment, simply run the following sequence of steps:

```
- bash fetch_and_preprocess.sh
- python main.py
```


### Acknowledgments
- The Gluon version is ported from this implementation [dasguptar/treelstm.pytorch](https://github.com/dasguptar/treelstm.pytorch)
- Shout-out to [Kai Sheng Tai](https://github.com/kaishengtai/) for the [original LuaTorch implementation](https://github.com/stanfordnlp/treelstm), and to the [Pytorch team](https://github.com/pytorch/pytorch#the-team) for the fun library.
