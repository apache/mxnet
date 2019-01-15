# Recommender Systems


This directory has a set of examples of how to build various kinds of recommender systems
using MXNet. The sparsity of user / item data is handled through the embedding layers that accept
indices as input rather than one-hot encoded vectors.


## Examples

The examples are driven by notebook files.

* [Matrix Factorization: linear and non-linear models](demo1-MF.ipynb)
* [Deep Structured Semantic Model (DSSM) for content-based recommendations](demo2-dssm.ipynb)


### Negative Sampling

* A previous version of this example had an example of negative sampling. For example of negative sampling, please refer to:
    [Gluon NLP Sampled Block](https://github.com/dmlc/gluon-nlp/blob/master/src/gluonnlp/model/sampled_block.py)
    

## Acknowledgements

Thanks to [xlvector](https://github.com/xlvector/) for the first Matrix Factorization example
that provided the basis for these examples.

[MovieLens](http://grouplens.org/datasets/movielens/) data from [GroupLens](http://grouplens.org/).
