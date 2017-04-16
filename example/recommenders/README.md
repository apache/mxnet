# Recommender Systems with Sparse Data

This directory has a set of examples of how to build various kinds of recommender systems
using MXNet.  It also includes a set of tools for using sparse data.

## Examples

The examples are driven by notebook files.

* [Matrix Factorization part 1: linear and non-linear models](demo1-MF.ipynb)
* [Matrix Factorization part 2: overfitting and deep ResNet](demo1-MF2-fancy.ipynb)
* [Binary classification with negative sampling](demo2-binary.ipynb)
* [Deep Structured Semantic Model (DSSM) for content-based recommendations](demo3-dssm.ipynb)

## Re-usable code

These examples use and demonstrate a number of layers and other tools that can be used outside of these examples.  They are all available from the [`recotools`](recotools.py) package.

### Negative Sampling

* `NegativeSamplingDataIter` 

### Loss Layers

* `CosineLoss`
* `CrossEntropyLoss`

### Sparse Data Projection layers

* `SparseRandomProjection`
* `SparseBagOfWordProjection`

## Acknowledgements

Thanks to [xlvector](https://github.com/xlvector/) for the first Matrix Factorization example
that provided the basis for these examples.

[MovieLens](http://grouplens.org/datasets/movielens/) data from [GroupLens](http://grouplens.org/).

