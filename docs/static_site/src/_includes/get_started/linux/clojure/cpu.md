You can use the Maven packages defined in the following dependency to include MXNet in your Clojure
project. To maximize leverage, the Clojure package has been built on the existing Scala package. Please
refer to the [MXNet-Scala setup guide]({{'/get_started/scala_setup'|relative_url}}) for a detailed set of instructions
to help you with the setup process that is required to use the Clojure dependency.

<a href="https://mvnrepository.com/artifact/org.apache.mxnet.contrib.clojure/clojure-mxnet-linux-cpu"><img
        src="https://img.shields.io/badge/org.apache.mxnet-linux cpu-green.svg"
        alt="maven badge"/></a>

{% highlight html %}
<dependency>
<groupId>org.apache.mxnet.contrib.clojure</groupId>
<artifactId>clojure-mxnet-linux-cpu</artifactId>
</dependency>
{% endhighlight %}
