# JVM Memory Management
The Scala and Java bindings of Apache MXNet uses native memory(C++ heap either in RAM or GPU memory) for most of the MXNet Scala objects such as NDArray, Symbol, Executor, KVStore, Data Iterators, etc.,
The Scala classes associated with them act as wrappers, the operations on these objects are directed to the MXNet C++ backend via JNI for performance, so the bytes are also stored in the native heap for fast access.

The JVM using the Garbage Collector only manages objects allocated in the JVM Heap and is not aware of the memory footprint of these objects in the native memory, hence allocation/deallocation of the native memory has to be managed by MXNet Scala.
Allocating native memory is straight forward and is done during the construction of the object by a calling the associated C++ API through JNI, However since JVM languages do not have destructors, deallocation of these objects needs to be done explicitly.
To make it easy, MXNet Scala provides a few modes of operation, explained in detail below.

## Memory Management in Scala 
### 1.  [ResourceScope.using](https://github.com/apache/incubator-mxnet/blob/master/scala-package/core/src/main/scala/org/apache/mxnet/ResourceScope.scala#L106) (Recommended)
`ResourceScope.using` provides the familiar Java try-with-resources primitive in Scala and also extends to automatically manage the memory of all the MXNet objects created in the code block (`body`) associated with it by tracking the allocations in a stack. 
If an MXNet object or an Iterable containing MXNet objects is returned from the code-block, it is automatically excluded from de-allocation in the current scope and moved to 
an outer scope if ResourceScope's are stacked.  

**Usage** 
```
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
In the example above, we have two ResourceScopes stacked together, 4 NDArrays `(r1, r2, r3, r4)` are created in the inner scope, the inner scope returns 
`(r3, r4)`. The ResourceScope code recognizes that it should not de-allocate these objects and automatically moves `r3` and  `r4` to the outer scope. The outer scope 
returns `r4` from its code-block and deallocates `r3`, so ResourceScope.using removes this from its list of objects to be de-allocated. All other objects are automatically released(native memory) by calling the C++ Backend to free the memory.

**Note:**
You should consider stacking ResourceScope when you have layers of functionality in your application code which creates a lot of MXNet objects like NDArray. 
This is because you don't want to hold onto all the memory that is created for the entire training loop and you will most likely run out of memory especially on GPUs which have limited memory in order 8 to 16 GB. 
For example if you were writing Training code in MXNet Scala, it is recommended not to use one-uber ResourceScope block that runs the entire training code, 
instead you should stack multiple scopes one where you run forward backward passes on each batch, 
and 2nd scope for each epoch and an outer scope that runs the entire training script, like the example below
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
When the Garbage Collector runs, GC identifies unreachable Scala/Java objects in the JVM Heap and finalizes them, 
we take advantage of Garbage Collector which enqueues objects into a reference queue that are ready to be reclaimed, 
at which point we do pre-mortem clean up by calling the corresponding MXNet backend API to free the native memory. 
 
In this approach, you do not have to write any special code to have native memory cleaned up, however this approach solely depends on the Garbage collector to run and find unreachable objects.
You can control the frequency of Garbage Collector by calling System.gc() at strategic points such as at the end of an epoch or at the end of a mini-batch in Training.

This approach could be suitable for use-cases such as inference on CPUs and you have large amount of Memory(RAM) on your system.  

**Note:**   
Calling GC too frequently can cause your application to perform poorly. This approach might not be suitable   
when you have large number of large NDArray allocations too quickly such as training a GAN model

### Using dispose Pattern (least Recommended)
 
There might be situations where you want to manage the lifecycle of Apache MXNet objects, for such use-cases we have provided `dispose()` method that you can call and it will deallocate associated native memory, we have also
made all MXNet objects [AutoCloseable](https://docs.oracle.com/javase/8/docs/api/java/lang/AutoCloseable.html), if you are using Java8 and above you can use it with try-with-resources or call close() in the finally block.

**Note:**   
We recommend to avoid manually managing MXNet objects and instead to use `ResourceScope.using` as this could leak memory if you miss calling dispose( at some point GC will kick in and be cleaned up due to Phantom Reference)
and create unreadable code.   

```
def showDispose(): Unit = {
    val r = NDArray.ones(Shape (2, 2))
    r.dispose()
}
```

## 3. Memory Management in Java
Memory Management in MXNet Java is similar to Scala, We recommend to use [ResourceScope](https://github.com/apache/incubator-mxnet/blob/master/scala-package/core/src/main/scala/org/apache/mxnet/ResourceScope.scala#L32) in a `try-with-resources` block or in a `try-finally` block.   
Java 7 onwards supports [try-with-resource](https://docs.oracle.com/javase/tutorial/essential/exceptions/tryResourceClose.html) where the resources declared in the try block are automatically closed. 
The above discussed ResourceScope implements AutoCloseable and tracks all MXNet Objects created at a Thread Local scope level. 

```
try(ResourceScope scope = new ResourceScope()) {
    NDArray test = NDArray.ones((Shape (2,2))
}
```
or 
```
try {
    ResourceScope scope = new ResourceScope()
    NDArray test = NDArray.ones((Shape(2,2))
} finally {
    scope.close()
}
``` 
**Note:**
ResourceScope within a try block tracks all MXNet Native Object Allocations (NDArray, Symbol, Executor, etc.,) and deallocates at
the end of the try block even the objects that are returned, ie., in the above even if `test` were to be returned the native memory associated
with it would be deallocated and if you use it outside of the try block, the process might crash due to illegal memory access.

If you want to retain certain objects created within the try block, you should explicitly remove them from the scope by calling `scope.moveToOuterScope`
It is highly recommended to use a stack of try-with-resource ResourceScope's so you do not have explicitly manage the lifecycle of the Native objects.

