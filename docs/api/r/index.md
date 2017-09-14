# MXNet - R API

See the [MXNet R Reference Manual](https://media.readthedocs.org/pdf/mxnet-test/latest/mxnet-test.pdf).

MXNet supports the R programming language. The MXNet R package brings flexible and efficient GPU
computing and state-of-art deep learning to R. It enables you to write seamless tensor/matrix computation with multiple GPUs in R. It also lets you construct and customize the state-of-art deep learning models in R,
  and apply them to tasks, such as image classification and data science challenges.

You can perform tensor or matrix computation in R:

```r
   > require(mxnet)
   Loading required package: mxnet
   > a <- mx.nd.ones(c(2,3))
   > a
        [,1] [,2] [,3]
   [1,]    1    1    1
   [2,]    1    1    1
   > a + 1
        [,1] [,2] [,3]
   [1,]    2    2    2
   [2,]    2    2    2
```
## Resources

* [MXNet R Reference Manual](https://media.readthedocs.org/pdf/mxnet-test/latest/mxnet-test.pdf)
* [MXNet for R Tutorials](../../tutorials/r/index.html)

