# MXNet compared to CXXNet
- The Symbolic Execution can be viewed as neural net execution(forward,
  backprop) with more optimizations.
- The Operator can be viewed as Layers, but need to pass in weights and bias.
	- It also contains more(optional) interface to further optimize memory usage.
- The Symbolic Construction module is advanced config file.
- The Runtime Dependency Engine engine is like a thread pool.
	- But makes your life easy to solve dependency tracking for you.
- KVStore adopts a simple parameter-server interface optimized for GPU
  synchronization.

# MXNet compared to Minerva
- The Runtime Dependency Engine is DAGEngine in Minerva, except that it is
  enhanced to support mutations.
- The NDArray is same as owl.NDArray, except that it supports mutation, and can
  interact with Symbolic Execution.
  
# Recommended Next Steps
* [MXNet Architecture] (http://mxnet.io/architecture/overview.html)
* [How to read MXNet code](http://mxnet.io/architecture/read_code.html)
* [Develop and hack MXNet](http://mxnet.io/how_to/develop_and_hack.html)