## SVRGModule Example

SVRGModule is an extension to the Module API that implements SVRG optimization, which stands for Stochastic
Variance Reduced Gradient. SVRG is an optimization technique that complements SGD and has several key
properties: 

* Employs explicit variance reduction by using a different update rule compared to SGD.
* Ability to use relatively large learning rate, which leads to faster convergence compared to SGD.
* Guarantees for fast convergence for smooth and strongly convex functions.

#### API Usage Example
SVRGModule provides both high-level and intermediate-level APIs while minimizing the changes with Module API.  
example_api_train.py: provides suggested usage of SVRGModule high-level and intermediate-level API.
example_inference.py: provides example usage of SVRGModule inference.

#### Linear Regression 
This example trains a linear regression model using SVRGModule on a real dataset, YearPredictionMSD. 
Logs of the training results can be  found in experiments.log which will automatically generated when running the 
training script.

##### Dataset
YearPredictionMSD: contains predictions of the release year of a song from audio features. It has over 
400,000 samples with 90 features. It will be automatically downloaded on first execution and cached.

YearPredictionMSD dataset: https://archive.ics.uci.edu/ml/datasets/yearpredictionmsd

#### Benchmarks:
An initial set of benchmarks has been performed on YearPredictionDatasetMSD with linear regression model.  A jupyter 
notebook under `/benchmarks` demonstrates the training process and plots two graphs for benchmarking.

* benchmark1: A lr_scheduler returns a new learning rate based on the number of updates that have been performed. 

* benchmark2: One drawback for SGD is that in order to converge faster, the learning rate has to decay to zero, 
thus SGD needs to start with a small learning rate. The learning rate does not need to decay to zero for SVRG, 
therefore we can use a relatively larger learning rate. SGD with learning rate of (0.001, 0.0025) and SVRG with 
learning rate of (0.025) are benchmarked. Even though SVRG starts with a relatively large learning rate, it converges 
much faster than SGD in both cases.  
