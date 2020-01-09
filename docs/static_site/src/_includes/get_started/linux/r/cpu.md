The default version of R that is installed with `apt-get` is insufficient. You will need
to first [install R v3.4.4+ and build MXNet from source](/get_started/ubuntu_setup.html#install-the-mxnet-package-for-r).

After you have setup R v3.4.4+ and MXNet, you can build and install the MXNet R bindings with the following, assuming that `incubator-mxnet` is the source directory you used to build MXNet as follows:

{% highlight bash %}
$ cd incubator-mxnet
$ mkdir build; cd build; cmake -DUSE_CUDA=OFF ..; make -j $(nproc); cd ..
$ make -f R-package/Makefile rpkg
{% endhighlight %}
