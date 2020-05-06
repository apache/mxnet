Note: packages for 3.6.x are not yet available.
Install 3.5.x of R from [CRAN](https://cran.r-project.org/bin/windows/base/old/).

You can [build MXNet-R from source](/get_started/windows_setup.html#install-the-mxnet-package-for-r), or you can use a pre-built binary:

{% highlight r %}
cran <- getOption("repos")
cran["dmlc"] <-
"https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/R/CRAN/GPU/cu92"
options(repos = cran)
install.packages("mxnet")
{% endhighlight %}

Change cu92 to cu90, cu91 or cuda100 based on your CUDA toolkit version. Currently, MXNet supports these versions of CUDA.
Note : You also need to have cuDNN installed on Windows. Check out this
[guide](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#installwindows)
on the steps for installation.