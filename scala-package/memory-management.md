# JVM Memory Management
The Scala and Java bindings of Apache MXNet use native memory (memory from the C++ heap in either RAM or GPU memory) for most of the MXNet objects such as NDArray, Symbol, Executor, KVStore, Data Iterators, etc.
The associated Scala classes act only as wrappers. The operations done on these wrapper objects are then directed to the high performance MXNet C++ backend via the Java Native Interface (JNI). Therefore, the bytes are stored in the C++ native heap which allows for fast access.

However, the JVM Garbage Collector only manages objects allocated in the JVM Heap and is not aware of the memory footprint of these objects in the native memory. Hence, the allocation/deallocation of native memory must be managed by MXNet Scala.
Allocating native memory is straight forward and is done during the construction of the object by calling the associated C++ API through JNI. However, since JVM languages do not have destructors, the deallocation of these objects must be done explicitly.
MXNet Scala provides a few easy modes of operation which are explained in detail below.

## Memory Management in Scala 
### 1.  [ResourceScope.using](https://github.com/apache/incubator-mxnet/blob/master/scala-package/core/src/main/scala/org/apache/mxnet/ResourceScope.scala#L106) (Recommended)
`ResourceScope.using` provides the familiar Java try-with-resources primitive in Scala and will automatically manage the memory of all the MXNet objects created in the associated code block (`body`). It works by tracking the allocations performed inside the code block deallocating when exiting the block. 
Passing MXNet objects out of a using block can be easily accomplished by simply returning an object or an iterable containing multiple MXNet objects. If you have nested using blocks, then the returned objects will be moved into the parent scope as well.

**Usage** 
```scala
ResourceScope.using() {
    ResourceScope.using() {
        val r1 = NDArray.ones(Shape(2, 2))
        val r2 = NDArray.ones(Shape(3, 4))
        val r3 = NDArray.ones(Shape(5, 6))
        val r4 = NDArray.ones(Shape(7, 8))
        (r3, r4)
    }
    r4
}
```
In the example above, we have two ResourceScopes stacked together. In the inner scope, 4 NDArrays `(r1, r2, r3, r4)` are created and the NDArrays 
`(r3, r4)` are returned. The inner ResourceScope recognizes that it should not deallocate these objects and automatically moves `r3` and  `r4` to the outer scope. When the outer scope 
returns `r4` from its code-block, it will only deallocate `r3` and will remove `r4` from its list of objects to be deallocated. All other objects are automatically released by calling the C++ backend to free the native memory.

**Note:**
You should consider nesting ResourceScopes when you have layers of functionality in your application code or create a lot of MXNet objects such as NDArrays.  
For example, holding onto all the memory that is created for an entire training loop can result in running out of memory, especially when training on GPUs which might only have 8 to 16 GB.  
It is recommended not to use a single ResourceScope block which spans the entire training code. You should instead nest multiple scopes: an innermost scope where you run forward-backward passes on each batch, a middle scope for each epoch, and an outer scope that runs the entire training script. This is demonstrated in the example below:

```scala
ResourceScope.using() {
 val m = Module()
 m.bind()
 val k = KVStore(...)
 ResourceScope.using() {
     val itr = MXIterator(..)
     val num_epochs: Int = 100
     //... 
     for (i <- 0 until num_epoch) {
     ResourceScope.using() {
        val dataBatch = itr.next()
        while(itr.next()) {
           m.forward(dataBatch)
           m.backward(dataBatch)
           m.update()
        }
     }
   }
 }
}

```  
       
### 2.  Using Phantom References (Recommended for some use cases)

Apache MXNet uses [Phantom References](https://docs.oracle.com/javase/8/docs/api/java/lang/ref/PhantomReference.html) to track all MXNet Objects that have native memory associated with it. 
When the Garbage Collector runs, it identifies unreachable Scala/Java objects in the JVM Heap and finalizes them. 
It then enqueues objects which are ready to be reclaimed into a reference queue. We take advantage of this and do a 
pre-mortem cleanup on these wrapper objects by freeing the corresponding native memory as well.
 
This approach is automatic and does not require any special code to clean up the native memory. However, the Garbage Collector is not aware of the potentially large amount of native memory used and therefore may not free up memory often enough with it's standard behavior.
You can control the frequency of garbage collection by calling System.gc() at strategic points such as the end of an epoch or the end of a mini-batch.

This approach could be suitable for some use cases such as inference on CPUs where you have a large amount of Memory (RAM) on your system.

**Note:**
Calling GC too frequently can also cause your application to perform poorly. This approach might not be suitable 
for use cases which quickly allocate a large number of large NDArrays such as when training a GAN model.

### 3. Using dispose Pattern (least Recommended)
 
There might be situations where you want to manually manage the lifecycle of Apache MXNet objects. For such use-cases, we have provided the `dispose()` method which will manually deallocate the associated native memory when called. We have also
made all MXNet objects [AutoCloseable](https://docs.oracle.com/javase/8/docs/api/java/lang/AutoCloseable.html). If you are using Java8 and above you can use it with try-with-resources or call close() in the finally block.

**Note:**
We recommend you avoid manually managing MXNet objects and instead use `ResourceScope.using`. This creates less readable code and could leak memory if you miss calling dispose (until it is cleaned up by the Garbage Collector through the Phantom References).

```scala
def showDispose(): Unit = {
    val r = NDArray.ones(Shape (2, 2))
    r.dispose()
}
```

## Memory Management in Java
Memory Management in MXNet Java is similar to Scala. We recommend you use [ResourceScope](https://github.com/apache/incubator-mxnet/blob/master/scala-package/core/src/main/scala/org/apache/mxnet/ResourceScope.scala#L32) in a `try-with-resources` block or in a `try-finally` block.
The [try-with-resource](https://docs.oracle.com/javase/tutorial/essential/exceptions/tryResourceClose.html) tracks the resources declared in the try block and automatically closes them upon exiting (supported from Java 7 onwards). 
The ResourceScope discussed above implements AutoCloseable and tracks all MXNet Objects created at a Thread Local scope level. 

```java
try(ResourceScope scope = new ResourceScope()) {
    NDArray test = NDArray.ones((Shape (2,2))
}
```
or 
```java
try {
    ResourceScope scope = new ResourceScope()
    NDArray test = NDArray.ones((Shape(2,2))
} finally {
    scope.close()
}
``` 

**Note:**
A ResourceScope within a try block tracks all MXNet Native Object Allocations (NDArray, Symbol, Executor, etc.,) and deallocates them at
the end of the try block. This is also true of the objects that are returned e.g. in the example above, the native memory associated with `test` would be deallocated even if it were to be returned. 
If you use the object outside of the try block, the process might crash due to illegal memory access.

To retain certain objects created within try blocks, you should explicitly remove them from the scope by calling `scope.moveToOuterScope`.
It is highly recommended to nest multiple try-with-resource ResourceScopes so you do not have to explicitly manage the lifecycle of the Native objects.

