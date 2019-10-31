Note: packages for 3.6.x are not yet available.
Install 3.5.x of R from [CRAN](https://cran.r-project.org/bin/windows/base/old/).

You can [build MXNet-R from source](/get_started/windows_setup.html#install-mxnet-package-for-r), or
you can use a
pre-built binary:

{% highlight r %}
cran <- getOption("repos")
cran["dmlc"] <- "https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/R/CRAN/"
options(repos = cran)
install.packages("mxnet")
{% endhighlight %}

To run MXNet you also should have OpenCV and OpenBLAS installed.
