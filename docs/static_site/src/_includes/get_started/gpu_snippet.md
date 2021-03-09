CUDA should be installed first. Starting from version 1.8.0, CUDNN and NCCL should be installed as well.

**Important:** Make sure your installed CUDA (CUDNN/NCCL if applicable) version matches the CUDA version in the pip package.  

Check your CUDA version with the following command:

{% highlight bash %}
nvcc --version
{% endhighlight %}

You can either upgrade your CUDA install or install the MXNet package that supports your CUDA version.