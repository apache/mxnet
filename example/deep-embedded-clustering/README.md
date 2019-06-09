# DEC Implementation
This is based on the paper `Unsupervised deep embedding for clustering analysis` by  Junyuan Xie, Ross Girshick, and Ali Farhadi

Abstract:

Clustering is central to many data-driven application domains and has been studied extensively in terms of distance functions and grouping algorithms. Relatively little work has focused on learning representations for clustering. In this paper, we propose Deep Embedded Clustering (DEC), a method that simultaneously learns feature representations and cluster assignments using deep neural networks. DEC learns a mapping from the data space to a lower-dimensional feature space in which it iteratively optimizes a clustering objective. Our experimental evaluations on image and text corpora show significant improvement over state-of-the-art methods.


## Prerequisite
  - Install Scikit-learn: `python -m pip install --user sklearn`
  - Install SciPy: `python -m pip install --user scipy`

## Data

The script is using MNIST dataset.

## Usage
run `python dec.py`
