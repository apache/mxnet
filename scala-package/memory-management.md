# JVM Memory Management
The Scala and Java bindings of Apache MXNet use native memory (C++ heap in either RAM or GPU memory) for most of the MXNet Scala objects such as NDArray, Symbol, Executor, KVStore, Data Iterators, etc.
The associated Scala classes act as wrappers. For performance, operations on these objects are directed to the MXNet C++ backend via JNI. Therefore, the bytes are also stored in the native heap allowing for fast access.

The JVM Garbage Collector only manages objects allocated in the JVM Heap and is not aware of the memory footprint of these objects in the native memory. Hence, the allocation/deallocation of native memory must be managed by MXNet Scala.
Allocating native memory is straight forward and is done during the construction of the object by calling the associated C++ API through JNI. However, since JVM languages do not have destructors, the deallocation of these objects needs to be done explicitly.
To make it easy, MXNet Scala provides a few modes of operation, explained in detail below.

## Memory Management in Scala 
### 1.  [ResourceScope.using](https://github.com/apache/incubator-mxnet/blob/master/scala-package/core/src/main/scala/org/apache/mxnet/ResourceScope.scala#L106) (Recommended)
`ResourceScope.using` provides the familiar Java try-with-resources primitive in Scala and is extended to automatically manage the memory of all the MXNet objects created in the associated code block (`body`). This is accomplished by tracking the allocations in a stack. 
An MXNet object, or iterable containing MXNet objects, is automatically excluded from deallocation when it is returned by the code block. If ResourceScopes are stacked then it will be added to the outer scope.

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
`(r3, r4)` are returned. The inner ResourceScope recognizes that it should not deallocate these objects and automatically moves `r3` and  `r4` to the outer scope. The outer scope 
returns `r4` from its code-block. The outer ResourceScope.using will deallocate `r3` and remove `r4` from its list of objects to be deallocated. All other objects are automatically released by calling the C++ backend to free the native memory.

**Note:**
You should consider stacking ResourceScope when you have layers of functionality in your application code which create a lot of MXNet objects like NDArray. 
This is because you don't want to hold onto all the memory that is created for an entire training loop, which could result in running out of memory (this is especially true on GPUs which have limited memory on the order of 8 to 16 GB). 
For example, if you were writing training code in MXNet Scala, it is recommended not to use a single ResourceScope block which spans the entire training code. 
Instead you should stack multiple scopes, one where you run forward backward passes on each batch, 
a 2nd scope for each epoch, and an outer scope that runs the entire training script. This is demonstrated in the example below:
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
When the Garbage Collector runs, GC identifies unreachable Scala/Java objects in the JVM Heap and finalizes them. 
The Garbage Collector enqueues objects which are ready to be reclaimed into a reference queue. We take advantage of this and do a 
pre-mortem cleanup on these objects by calling the corresponding MXNet backend API to free the native memory.
 
In this approach, you do not have to write any special code to have native memory cleaned up. However, this approach solely depends on the Garbage collector to run and find unreachable objects.
You can control the frequency of Garbage Collector by calling System.gc() at strategic points, such as at the end of an epoch or at the end of a mini-batch in training.

This approach could be suitable for use-cases such as inference on CPUs and you have large amount of Memory(RAM) on your system.  

**Note:**   
Calling GC too frequently can cause your application to perform poorly. This approach might not be suitable 
for use cases which quickly allocate a large number of large NDArrays, such as when training a GAN model.

### Using dispose Pattern (least Recommended)
 
There might be situations where you want to manually manage the lifecycle of Apache MXNet objects. For such use-cases, we have provided the `dispose()` method which will deallocate the associated native memory when called. We have also
made all MXNet objects [AutoCloseable](https://docs.oracle.com/javase/8/docs/api/java/lang/AutoCloseable.html). If you are using Java8 and above you can use it with try-with-resources or call close() in the finally block.

**Note:**   
We recommend to avoid manually managing MXNet objects and instead to use `ResourceScope.using`. As this could leak memory if you miss calling dispose (at some point GC will kick in and be cleaned up due to Phantom Reference)
and creates less readable code.   

```scala
def showDispose(): Unit = {
    val r = NDArray.ones(Shape (2, 2))
    r.dispose()
}
```

## 3. Memory Management in Java
Memory Management in MXNet Java is similar to Scala. We recommend to use [ResourceScope](https://github.com/apache/incubator-mxnet/blob/master/scala-package/core/src/main/scala/org/apache/mxnet/ResourceScope.scala#L32) in a `try-with-resources` block or in a `try-finally` block.   
Java 7 onwards supports [try-with-resource](https://docs.oracle.com/javase/tutorial/essential/exceptions/tryResourceClose.html) where the resources declared in the try block are automatically closed. 
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
ResourceScope within a try block tracks all MXNet Native Object Allocations (NDArray, Symbol, Executor, etc.,) and deallocates at
the end of the try block. This is also true of the objects that are returned, i.e., in the example above the native memory associated with `test` would be deallocated even if it were to be returned. 
If you use it outside of the try block, the process might crash due to illegal memory access.

If you want to retain certain objects created within the try block, you should explicitly remove them from the scope by calling `scope.moveToOuterScope`.
It is highly recommended to use a stack of try-with-resource ResourceScope's so you do not have explicitly manage the lifecycle of the Native objects.

