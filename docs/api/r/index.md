# MXNet - R API
<script type="text/javascript" src='../../_static/js/versions_drop_down.js'></script>
  <div class="dropdown">
    <button class="btn current-version btn-primary dropdown-toggle" type="button" data-toggle="dropdown">v1.2.0
    <span class="caret"></span></button>
    <ul class="dropdown-menu opt-group">
      <li class="opt active versions"><a href="#">v1.2.0</a></li>
      <li class="opt versions"><a href="#">v1.1.0</a></li>
      <li class="opt versions"><a href="#">v1.0.0</a></li>
      <li class="opt versions"><a href="#">v0.12.1</a></li>
      <li class="opt versions"><a href="#">v0.11.0</a></li>
      <li class="opt versions"><a href="#">master</a></li>
    </ul>
  </div>


See the [MXNet R Reference Manual](https://s3.amazonaws.com/mxnet-prod/docs/R/mxnet-r-reference-manual.pdf).

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

* [MXNet R Reference Manual](https://s3.amazonaws.com/mxnet-prod/docs/R/mxnet-r-reference-manual.pdf)
* [MXNet for R Tutorials](../../tutorials/r/index.html)
