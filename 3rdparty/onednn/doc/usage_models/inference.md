Inference {#dev_guide_inference}
================================

oneDNN includes primitives for operations throughout a deep learning
network topology. However, it is important to note the scope of oneDNN
is limited to performance critical functionality and the library does not
provide all the functions necessary to implement deep learning workloads, for
instance data preprocessing or computing loss function.  The soft-max
classifier is the sole classifier included, but the application of other
classifier types will require user's own implementations. The scope of the
library is depicted in the following image:

@img{img_inference_scope.jpg,,80%,}

## Best Practices for Inference

## fp32 Inference

**Use Forward Inference Primitives**

oneDNN provides a forward pass
version of each primitive, that avoids storing information required for a
backward pass (as in training).

Use the #dnnl::prop_kind::forward_inference argument at creation of the
**operation descriptor**, as in this convolution example:
~~~cpp
auto conv_descr = convolution_forward::desc(prop_kind::forward_inference, ...);
~~~

**Layout Propagation**

Compute-intensive oneDNN primitives execute with highest performance
on CPU-friendly data formats. Please see description of data formats
[here](@ref memory_format_propagation_cpp).

Performance gains are maximized by reordering once, and then propagating the
CPU-friendly format through as many layers as possible in your topology.
oneDNN provides the `format_tag=any` for memory descriptors that will be
passed to compute-intensive primitives. The compute-intensive primitive types
in oneDNN are @ref dev_guide_convolution, @ref dev_guide_inner_product,
and @ref dev_guide_rnn.

To accomplish this propagation in a robust manner, its is recommended to
follow these steps:

A. On compute-intensive operations:
* Pass the `format_tag=any` when creating oneDNN **memory descriptor**
  for source, destination, and weights memory
* Use these three *memory descriptors* with 'format _tag=any` to create
  **operation descriptor**
* Use *operation descriptor* to create engine-aware **primitive descriptor**
* Query the *primitive descriptor* with `.src_desc()` method to get recommended
  format
* Write conditional reorder to execute only if user source data or weights
  don't match the recommended format
* Create **primitive** and add it to stream
  with `primitive.execute(stream, args)`

B. On non-intensive operations:
* Query output **primitive descriptor** with `.dst_desc()` from previous
  operation to find current layout
* Pass current layout with `format_tag=.dst_desc()` when creating non-intensive
  **operation descriptor**
* Create **primitive** and add it to stream
  with `operation.execute(stream, args)`

Now let's take a look at the code syntax to accomplish the compute-intensive
steps:

Pass the `format_tag=any` when creating oneDNN **memory descriptor**
for source, destination, and weights memory
~~~cpp
source_mem_descr = memory::desc(args*, memory::format_tag::any);
dest_mem_descr = memory::desc(args*, memory::format_tag::any);
weights_mem_descr = memory::desc(args*, memory::format_tag::any);
~~~

Use these three *memory descriptors* with 'format _tag=any`
to create **operation descriptor**
~~~cpp
auto conv_descr = convolution_forward::desc(...,
            source_mem_descr, weights_mem_descr, dest_mem_descr);
~~~

Use *operation descriptor* to create engine-aware **primitive descriptor**
~~~cpp
auto conv_prim_descr = convolution_forward::primitive_desc(conv_descr, engine);
~~~

Query the *primitive descriptor* with `.src_desc()` method to get recommended
format
Write conditional reorder to execute only if user source data or weights
don't match the recommended format
(Note: Do this for weight_memory as well)
~~~cpp
memory conv_source_memory = user_source_memory;
if (conv_prim_descr.src_desc() != user_source_memory.get_desc()) {
    conv_source_memory = memory(conv_prim_descr.src_desc(), engine);
    auto reorder_prim_descr = reorder::primitive_desc(user_source_memory, conv_source_memory);
    reorder(reorder_prim_descr).execute(s, user_source_memory, conv_source_memory);
}
~~~
Create **primitive** and add it to stream with `primitive.execute(stream, args)`
~~~cpp
auto conv = convolution_forward(conv_prim_descr);
conv.execute(s, {
            {DNNL_ARG_SRC, conv_source_memory},
            {DNNL_ARG_WEIGHTS, conv_weights_memory},
            {DNNL_ARG_DST, conv_dest_memory}});
~~~


**Cache Weights**\
Weights are accessed many times during batched inference. At inference time
these weights are essentially constants in the mapping function that the network
is applying to the input data. As such, the weights should be reordered
(if necessary) once and then used in the reorder form for the duration
of the execution. This caching causes the computer to use them in a way similar
to how a mathematical function applies a constant, i..e, "Grab-and-go"
with no overhead for creation or reorder.

**Primitive Reuse**\
There is JIT compilation overhead associated with primitive creation. It is
recommended to reuse any primitive that you can, and only create them once.

**Fused Primitives**\
oneDNN provides fused versions of primitives that attach a non-intensive
operation to the end of a compute-intensive operation and then executes both
in a single pass, reducing the number of memory accesses needed
for the combined operations.
The non-intensive operation is added as a **post-op** attribute to the compute
intensive primitive descriptor. Please note that post-ops do not change
the number of inputs or outputs of the primitives. Please see
the "Post-ops and Attributes" section of the doc for each primitive type
in /docs/primitive/ for a list of available fused primitives.

A good example is adding ReLU as a post-op to convolution, which we will use
as a demonstration below. The steps are

* Create a `post_op` for fused ReLU
* Create **primitive attribute** and add the `post_op`
* Create a convolution **descriptor**
* Create a convolution **primitive descriptor**, passing `post_op as` an arg

Create a `post_op` for fused ReLU
~~~cpp
post_ops ops;
ops.append_eltwise(..., algorithm::eltwise_relu);
~~~

Create **primitive attribute** and add the `post_op`
~~~cpp
primitive_attr attr;
attr.set_post_ops(ops);
~~~

Create a convolution **descriptor**
~~~cpp
auto conv_descr = convolution_forward::desc(...);
~~~

Create a convolution **primitive descriptor**, passing the post-op infused
`attrs` as an arg
~~~cpp
auto conv_prim_descr = convolution_forward::primitive_desc(conv_descr, attrs, engine);
~~~

## int8 Inference

oneDNN supports low precision int8 for inference execution. Note that not all
 primitives have int8 versions. Sometimes the speed benefits would be minimal,
or the loss in accuracy is not acceptable. Also the soft-max classifier only
supports fp32, so int8 inference will require a reorder before executing this
primitive.

By default, the oneDNN reorder primitive does not scale upon casting to int8.
In order to compress fp32 data to int8 precision while still preserving
the entire shape of the distribution, a process called **quantization** must
applied. Quantization will scale the data based on its range to efficiently fill
the bits available for int8 type.

To achieve quantization upon casting, the user must provide a few inputs to
oneDNN in order to use int8 inference:

* Specify data type at creation of primitive descriptor (int8 in this case)
* Provide a scaling factor for oneDNN reorder primitive
* Provide an output scaling factor the operation primitive

Please see the dedicated [section](@ref dev_guide_inference_int8) on low
precision computations in oneDNN for a detailed discussion, including how
to calculate the scaling factors.
