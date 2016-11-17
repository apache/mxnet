Bayesian Methods
================

This folder contains examples related to Bayesian Methods.

We currently have *Stochastic Gradient Langevin Dynamics (SGLD)* [<cite>(Welling and Teh, 2011)</cite>](http://www.icml-2011.org/papers/398_icmlpaper.pdf)
and *Bayesian Dark Knowledge (BDK)* [<cite>(Balan, Rathod, Murphy and Welling, 2015)</cite>](http://papers.nips.cc/paper/5965-bayesian-dark-knowledge).

**sgld.ipynb** shows how to use MXNet to repeat the toy experiment in the original SGLD paper.

**bdk.ipynb** shows how to use MXNet to implement the DistilledSGLD algorithm in Bayesian Dark Knowledge.

**bdk_demo.py** contains scripts (more than the notebook) related to Bayesian Dark Knowledge. Use `python bdk_demo.py -d 1 -l 2 -t 50000` to run classification on MNIST. 
