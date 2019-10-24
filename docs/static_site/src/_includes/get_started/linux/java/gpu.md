You can use the Maven packages defined in the following dependency to include MXNet in your Java
project. The Java API is provided as a subset of the Scala API and is intended for inference only.
Please refer to the <a href="/get_started/java_setup.html">MXNet-Java setup guide</a> for a detailed set of
instructions to help you with the setup process.

<a href="https://repository.apache.org/#nexus-search;gav~org.apache.mxnet~~1.5.0~~">
    <img src="https://img.shields.io/badge/org.apache.mxnet-linux gpu-green.svg"
    alt="maven badge"/>
</a>

{% highlight html %}
<dependency>
    <groupId>org.apache.mxnet</groupId>
    <artifactId>mxnet-full_2.11-linux-x86_64-gpu</artifactId>
    <version>[1.5.0, )</version>
</dependency>
{% endhighlight %}