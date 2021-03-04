Primitive Cache {#dev_guide_primitive_cache}
===========================================================

Primitive creation time largely depends on the underlying implementation,
for instance, oneDNN uses just-in-time compilation (JIT) to generate optimal
code for some CPU and GPU implementations, which introduces overhead.

To mitigate primitive creation overhead, oneDNN provides the primitive cache
which automatically caches created primitives to avoid repeating JIT compilation
for the primitives with identical operation descriptors, attributes, underlying
primitive implementations, etc. It can significantly reduce primitive creation
overhead, especially when an application or a framework creates primitives
for every instance of inference or iteration of training process.

The primitive cache is global hence a user does not have to maintain any
persistent oneDNN resources to benefit from the primitive cache.

## Managing Memory Consumption
The primitive cache has an upper limit for the number of primitives stored. Once
capacity is exceeded, a primitive that was least recently used will be evicted
from the cache. See the Run-time Controls section below for information on
changing the cache capacity.

## Profiling
Information about primitive cache hits and misses can be used for debug
purposes. That information is part of the verbose output for verbose
level 2 (@ref dev_guide_verbose).

## Build-time Controls

At build-time, support for this feature is controlled via cmake option
`DNNL_ENABLE_PRIMITIVE_CACHE`.

| CMake Option                | Supported values (defaults in bold) | Description
| :---                        | :---                                | :---
| DNNL_ENABLE_PRIMITIVE_CACHE | **ON**, OFF                         | Enables [primitive cache](@ref dev_guide_primitive_cache)

## Run-time Controls
When the feature is enabled at build-time, the `DNNL_PRIMITIVE_CACHE_CAPACITY`
environment variable can be used to change cache capacity or disable the cache.

| Environment variable          | Value            | Description
| :---                          | :---             | :---
| DNNL_PRIMITIVE_CACHE_CAPACITY | \<number\>       | Set cache capacity to \<number\> (default **1024**)
|                               | 0                | Disable primitive cache

This feature can also be managed at run-time with the following functions:
* @ref dnnl_set_primitive_cache_capacity

The function setting takes precedence over the environment variable.
