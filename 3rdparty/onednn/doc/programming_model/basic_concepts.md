Basic Concepts {#dev_guide_basic_concepts}
==========================================

## Introduction

In this page, an outline of the oneDNN programming model is presented, and the
key concepts are discussed, including *Primitives*, *Engines*, *Streams*, and
*Memory Objects*. In essence, the oneDNN programming model consists in executing
one or several *primitives* to process data in one or several *memory objects*.
The execution is performed on an *engine* in the context of a *stream*. The
relationship between these entities is briefly presented in Figure 1, which also
includes additional concepts relevant to the oneDNN programming model, such as
primitive *attributes* and *descriptors*. These concepts are described below in
much more details.

@img{img_programming_model.png,Figure 1: Overview of oneDNN programming model. Blue rectangles denote oneDNN objects\, and red lines denote dependencies between objects.,60%,}

### Primitives

oneDNN is built around the notion of a *primitive* (@ref dnnl::primitive). A
*primitive* is a functor object that encapsulates a particular computation such
as forward convolution, backward LSTM computations, or a data transformation
operation. Additionally, using primitive *attributes* (@ref
dnnl::primitive_attr) certain primitives can represent more complex *fused*
computations such as a forward convolution followed by a ReLU.

The most important difference between a primitive and a pure function is that
a primitive can store state.

One part of the primitive’s state is immutable. For example, convolution
primitives store parameters like tensor shapes and can pre-compute other
dependent parameters like cache blocking. This approach allows oneDNN primitives
to pre-generate code specifically tailored for the operation to be performed.
The oneDNN programming model assumes that the time it takes to perform the
pre-computations is amortized by reusing the same primitive to perform
computations multiple times.

The mutable part of the primitive’s state is referred to as a scratchpad. It
is a memory buffer that a primitive may use for temporary storage only during
computations. The scratchpad can either be owned by a primitive object (which
makes that object non-thread safe) or be an execution-time parameter.

### Engines

*Engines* (@ref dnnl::engine) is an abstraction of a computational device: a
CPU, a specific GPU card in the system, etc. Most primitives are created to
execute computations on one specific engine. The only exceptions are reorder
primitives that transfer data between two different engines.

### Streams

*Streams* (@ref dnnl::stream) encapsulate execution context tied to a
particular engine. For example, they can correspond to OpenCL command queues.

### Memory Objects

*Memory objects* (@ref dnnl::memory) encapsulate handles to memory allocated
on a specific engine, tensor dimensions, data type, and memory format – the
way tensor indices map to offsets in linear memory space. Memory objects are
passed to primitives during execution.

## Levels of Abstraction

oneDNN has multiple levels of abstractions for primitives and memory objects
in order to expose maximum flexibility to its users.

On the *logical* level, the library provides the following abstractions:

* *Memory descriptors* (@ref dnnl::memory::desc) define a tensor's logical
  dimensions, data type, and the format in which the data is laid out in
  memory. The special format _any_ (@ref dnnl::memory::format_tag::any)
  indicates that the actual format will be defined later (see @ref
  memory_format_propagation_cpp).

* *Operation descriptors* (one for each supported primitive) describe an
  operation's most basic properties without specifying, for example, which
  engine will be used to compute them. For example, convolution descriptor
  describes shapes of source, destination, and weights tensors, propagation
  kind (forward, backward with respect to data or weights), and other
  implementation-independent parameters.

* *Primitive descriptors* (@ref dnnl_primitive_desc_t; in the C++ API there
  are multiple types for each supported primitive) are at an abstraction level
  in between operation descriptors and primitives and can be used to inspect
  details of a specific primitive implementation like expected memory formats
  via queries to implement memory format propagation (see @ref
  memory_format_propagation_cpp) without having to fully instantiate a
  primitive.


| Abstraction level        | Memory object     | Primitive objects    |
|--------------------------|-------------------|----------------------|
| Logical description      | Memory descriptor | Operation descriptor |
| Intermediate description | N/A               | Primitive descriptor |
| Implementation           | Memory object     | Primitive            |

## Creating Memory Objects and Primitives

### Memory Objects

Memory objects are created from the memory descriptors. It is not possible to
create a memory object from a memory descriptor that has memory format set to
#dnnl::memory::format_tag::any.

There are two common ways for initializing memory descriptors:

* By using @ref dnnl::memory::desc constructors or by extracting a
  descriptor for a part of a tensor via
  @ref dnnl::memory::desc::submemory_desc

* By *querying* an existing primitive descriptor for a memory descriptor
  corresponding to one of the primitive's parameters (for example, @ref
  dnnl::convolution_forward::primitive_desc::src_desc).

Memory objects can be created with a user-provided handle (a `void *` on CPU),
or without one, in which case the library will allocate storage space on its
own.

### Primitives

The sequence of actions to create a primitive is:

1. Create an operation descriptor via, for example, @ref
   dnnl::convolution_forward::desc. The operation descriptor can contain
   memory descriptors with placeholder
   [format_tag::any](@ref dnnl::memory::format_tag::any)
   memory formats if the primitive supports it.

2. Create a primitive descriptor based on the operation descriptor, engine
   and attributes.

3. Create a primitive based on the primitive descriptor obtained in step 2.

@note The above sequence does not relate to all primitives in its entirety. For
instance, the reorder primitive does not have an operation descriptor.
