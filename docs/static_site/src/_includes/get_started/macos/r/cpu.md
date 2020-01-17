To run MXNet you also should have OpenCV and OpenBLAS installed. You may install them with `brew` as follows:

{% highlight bash %}
brew install opencv
brew install openblas
{% endhighlight %}

To ensure MXNet R package runs with the version of OpenBLAS installed, create a symbolic link as follows:

{% highlight bash %}
ln -sf /usr/local/opt/openblas/lib/libopenblas.dylib
/usr/local/opt/openblas/lib/libopenblasp-r0.3.1.dylib
{% endhighlight %}

Note: packages for 3.6.x are not yet available.

Install 3.5.x of R from [CRAN](https://cran.r-project.org/bin/macosx/). The latest is
[v3.5.3](https://cran.r-project.org/bin/macosx/R-3.5.3.pkg).

You can [build MXNet-R from source](get_started/osx_setup.html#install-the-mxnet-package-for-r), or
you can use a pre-built binary:

{% highlight r %}
cran <- getOption("repos")
cran["dmlc"] <- "https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/R/CRAN/"
options(repos = cran)
install.packages("mxnet")
{% endhighlight %}
