Primitive Attributes: Quantization {#dev_guide_attributes_quantization}
=======================================================================

@anchor dgaq_intro
## Introduction

Some primitives in the library support input/output tensors with the INT8
(either signed or unsigned) data type. The primary goal is to support
reduced precision inference on the compatible hardware.

Related materials:
- [Lower Numerical Precision Deep Learning Inference and Training](https://software.intel.com/content/www/us/en/develop/articles/lower-numerical-precision-deep-learning-inference-and-training)
- An example with annotations: @ref dev_guide_inference_int8

## Quantization Model

The primary quantization model that the library assumes is the following:
\f[
    x_{f32}[:] = scale_{f32} \cdot (x_{int8}[:] - 0_{x\_int8})
\f]

where \f$scale_{f32}\f$ is a *scaling factor* that is somehow known in advance
and \f$[:]\f$ is used to denote elementwise application of the formula to the
arrays. Typically, the process of obtaining these scale factors is called the
*calibration*. This might be counter-intuitive, but the library cannot compute
any of the scale factors at run-time dynamically. Hence, the model is
sometimes called a *static* quantization model. The main rationale to support
only *static* quantization out-of-the-box is higher performance. To use
*dynamic* quantization:
1. Compute the result in higher precision, like #dnnl::memory::data_type::s32.
2. Find the required characteristics, like min and max values, and derive
   the scale factor.
3. Re-quantize to the lower precision data type.

It is also worth mentioning that the library supports fixed zero position.
For most of the primitives, real zero value is mapped to zero for quantized
values; that is, \f$0_{x\_int8} = 0\f$. For example, this is the only model that
@ref dev_guide_convolution and @ref dev_guide_inner_product currently support.
The @ref dev_guide_rnn primitives have limited support of shifted zero (for
details, refer to the corresponding section in @ref dev_guide_rnn).

For the rest of this guide, we will assume that \f$0_{x\_int8} = 0\f$.

@warning
Depending on the architecture, the behavior of int8 computations might slightly
vary. For more details, refer to @ref dev_guide_int8_computations.

This guide does not cover how the appropriate scaling factor can be found.
Refer to the materials in the [Introduction](@ref dgaq_intro).

### Example: Convolution Quantization Workflow

Consider a convolution without bias. The tensors are represented as:

- \f$\src_{f32}[:] = scale_{\src} \cdot \src_{int8}[:]\f$
- \f$\weights_{f32}[:] = scale_{\weights} \cdot \weights_{int8}[:]\f$
- \f$\dst_{f32}[:] = scale_{\dst} \cdot \dst_{int8}[:]\f$

Here the \f$\src_{f32}, \weights_{f32}, \dst_{f32}\f$ are not
computed at all, the whole work happens with INT8 tensors.
As mentioned above, we also somehow know all the scaling factors:
\f$scale_{\src}, scale_{\weights}, scale_{\dst}\f$.

So the task is to compute the \f$\dst_{int8}\f$ tensor.

Mathematically, the computations are straightforward:
\f[
    \dst_{int8}[:] =
        downconvert\_f32\_to\_int8(
            output\_scale \cdot
            conv_{s32}(\src_{int8}, \weights_{int8})
        ),
\f]

where
- \f$output\_scale := \frac{scale_{\src} \cdot scale_{\weights}}{scale_{\dst}}\f$;
- \f$conv_{s32}\f$ is just a regular convolution which takes source and
  weights with INT8 data type and compute the result in INT32 data type (INT32
  is chosen to avoid overflows during the computations);
- \f$downconvert\_f32\_to\_s8()\f$ converts an `f32` value to `s8` with
  potential saturation if the values are out of the range of the INT8 data type.

Note that in order to perform the operation, one does not need to know the
exact scaling factors for all the tensors; it is enough to know only the
\f$output\_scale\f$. The library utilizes this fact: a user needs to provide
only this one extra parameter to the convolution primitive (see the [Output
Scaling Attribute](@ref dev_guide_attributes_quantization_output_scare)
section below).

### Per-Channel Scaling

Some of the primitives have limited support of multiple scales for a quantized
tensor. The most popular use case is the @ref dev_guide_convolution primitive
that supports per-output-channel scaling factors for the weights, meaning that
the actual convolution computations would need to scale different output
channels differently. This is possible without significant performance loss
because the per-output-channel re-quantization is only required at the very end
of the computations. It seems impossible to implement the same trick for the
input channels, since that would require re-quantization for every input
data point.

Let \f$\alpha\f$ denote scales:
- \f$\src_{f32}(n, ic, ih, iw) = \alpha_{\src} \cdot \src_{int8}(n, ic, ih, iw)\f$
- \f$\weights_{f32}(oc, ic, kh, kw) =
    \alpha_{\weights}(oc) \cdot \weights_{int8}(oc, ic, kh, kw)\f$
- \f$\dst_{f32}(n, oc, oh, ow) = scale_{\dst} \cdot \dst_{int8}(n, oc, oh, ow)\f$

Note that now the weights' scaling factor depends on the \f$oc\f$.

To compute the \f$\dst_{int8}\f$ we need to perform the following:

\f[
    \dst_{int8}(n, oc, oh, ow) =
        downconvert\_f32\_to\_int8(
            output\_scale(oc) \cdot
            conv_{s32}(\src_{int8}, \weights_{int8})|_{(n, oc, oh, ow)}
        ),
\f]

where
- \f$output\_scale(oc) :=
    \frac{\alpha_{\src} \cdot \alpha_{\weights}(oc)}{\alpha_{\dst}}\f$.

It is worth mentioning that the user is responsible for preparing quantized
weights accordingly. oneDNN provides reorders that can perform per-channel
scaling:

\f[
    \weights_{int8}(oc, ic, kh, kw) =
        downconvert\_f32\_to\_int8(
            output\_scale(oc) \cdot
            \weights_{f32}(oc, ic, kh, kw)
        ),
\f]

where
- \f$output\_scale(oc) := \frac{1}{\alpha_{\weights}(oc_{})}\f$.


## API

The library API to support for INT8 was designed for the model described above.
However, it does not require users to follow exactly this model. As long as
users can fit their model into the given functionality everything should work
fine. Having this in mind we tried to design a minimal and simple yet powerful
enough quantization API.

The most common data type for data tensors during INT8 inference is
 #dnnl::memory::data_type::s8 and #dnnl::memory::data_type::u8. All the
scaling factors related to tensors are not attached in any way to the
oneDNN memory objects and should be maintained by users.

The library essentially extends the ability of the primitives to scale the
output before storing the result to the memory with the destination data type.
That's exactly the minimum that we need to support INT8 inference (check the
equations above--only \f$output\_scale\f$ is non-standard).

The scaling happens in the single precision floating point data type
(#dnnl::memory::data_type::f32). Before storing, the result is downconverted
to the destination data type with saturation if required. The rounding happens
according to the current HW setting (for instance, on CPU according to the
MXCSR register).


@anchor dev_guide_attributes_quantization_output_scare
### Output Scaling Attribute

The library uses @ref dev_guide_attributes API for setting the scaling factors
for most of the primitives. The supporting attributes can be found in the
documentation for each primitive. The unsupported cases are handled according
to the
[attributes error handling section](@ref dev_guide_attributes_error_handling).

API:
- C: @ref dnnl_primitive_attr_set_output_scales
- C++: @ref dnnl::primitive_attr::set_output_scales

The primitives do not support output scales if source (and weights) tensors
are not of the int8 data type. In other words, regular `f32` convolution cannot
scale the output result.

The parameters (C++ API for simplicity):
~~~cpp
void dnnl::primitive_attr::set_output_scales(
        int mask,
        const std::vector<float> &scales
        );
~~~

In the simplest case, when there is only one common scale the attribute changes
the op behavior from
\f[
    \dst[:] = Op(...)
\f]

to

\f[
    \dst[:] = scale \cdot Op(...).
\f]

To support scales per one or several dimensions, users must set the appropriate
mask.

Say the destination is a \f$D_0 \times ... \times D_{n-1}\f$ tensor and
we want to have output scales per \f$d_i\f$ dimension
(where \f$0 \le d_i < n\f$).

Then the mask should be set to:
- \f$mask = \sum \limits_{d_i} 2^{d_i}\f$,

and the number of scales should be:
- `scales.size()` = \f$\prod\limits_{d_i}D_{d_i}\f$.

#### Example 1: weights quantization with per-output-channel-and-group scaling

~~~cpp
// weights dimensions
const int G, OC, IC, KH, KW;

// original f32 weights in user's format
dnnl::memory::desc wei_user_f32_md(
        {G, OC/G, IC/G, KH, KW},            // dims
        dnnl::memory::data_type::f32,     // the data originally in f32
        dnnl::memory::format_tag::hwigo   // the memory format a user uses
        );

// the scaling factors for quantized weights
// An unique scale for each group and output-channel.
std::vector<float> wei_scales(G * OC/G) = {...};

// ...

// int8 convolution primitive descriptor (will create it in the next example)
dnnl::convolution_forward::primitive_desc conv_pd(...);


// query the convolution weights memory descriptor
dnnl::memory::desc wei_conv_s8_md = conv_pd.weights_desc();

// prepare the inverse of the scales (f32 = scale * int8 --> int8 = 1/scale * f32)
std::vector<float> inv_wei_scales(wei_scales.size());
for (size_t i = 0; i < wei_scales.size(); ++i)
    inv_wei_scales[i] = 1.f / wei_scales[i];

// prepare the attributes for the reorder
dnnl::primitive_attr attr;
const int mask = 0
    | (1 << 0)  // scale per  G dimension, which is the dim #0
    | (1 << 1); // scale per OC dimension, which is the dim #1
attr.set_output_scales(mask, inv_wei_scales);

// create reorder that would perform:
//   wei_s8(g, oc, ic, kh, kw) <- 1/scale(g, oc) * wei_f32(g, oc, ic, kh, kw)
// including the data format transformation.
auto wei_reorder_pd = dnnl::reorder::primitive_desc(
        wei_user_f32_md, engine, // source
        wei_conv_s8_md, engine, // destination,
        attr);
auto wei_reorder = dnnl::reorder(wei_reorder_pd);

// ...
~~~

#### Example 2: convolution with groups, with per-output-channel quantization

This example is complementary to the previous example (which should ideally
be the first one). Let's say we want to have an INT8 convolution with
per-output channel scaling.

~~~cpp
const float src_scale; // src_f32[:] = src_scale * src_s8[:]
const float dst_scale; // dst_f32[:] = dst_scale * dst_s8[:]

// the scaling factors for quantized weights (as declared above)
// An unique scale for each group and output-channel.
std::vector<float> wei_scales(G * OC/G) = {...};


// Src, weights, and dst memory descriptors for convolution,
// with memory format tag == any to allow a convolution implementation
// to chose the appropriate memory format

dnnl::memory::desc src_conv_s8_any_md(
        {BATCH, IC, IH, IW},            // dims
        dnnl::memory::data_type::s8,  // the data originally in s8
        dnnl::memory::format_tag::any // let convolution to choose
        );

dnnl::memory::desc wei_conv_s8_any_md(
        {G, OC/G, IC/G, KH, KW},        // dims
        dnnl::memory::data_type::s8,  // the data originally in s8
        dnnl::memory::format_tag::any // let convolution to choose
        );

dnnl::memory::desc dst_conv_s8_any_md(...);  // ditto

// Create a convolution operation descriptor
dnnl::convolution_forward::desc conv_d(
        dnnl::prop_kind::forward_inference,
        dnnl::algorithm::convolution_direct,
        src_conv_s8_any_md,                     // what's important is that
        wei_conv_s8_any_md,                     // we specified that we want
        dst_conv_s8_any_md,                     // computations in s8
        strides, padding_l, padding_r,
        dnnl::padding_kind::zero
        );

// prepare the attributes for the convolution
dnnl::primitive_attr attr;
const int mask = 0
    | (1 << 1); // scale per OC dimension, which is the dim #1 on dst tensor:
                // (BATCH, OC, OH, OW)
                //    0     1   2   3
std::vector<float> conv_output_scales(G * OC/G);
for (int g_oc = 0; G * OC/G; ++g_oc)
    conv_output_scales[g_oc] = src_scale * wei_scales(g_oc) / dst_scale;
attr.set_output_scales(mask, conv_output_scales);

// create a convolution primitive descriptor with the scaling factors
auto conv_pd = dnnl::convolution_forward::primitive_desc(
        conv_d, // general (non-customized) operation descriptor
        attr,   // the attributes contain the output scaling
        engine);

// ...
~~~

#### Interplay of output scales with post-ops

In general, the [post-ops](@ref dev_guide_attributes_post_ops) are independent
from the output scales. The output scales are applied to the result first; then
post-ops will take effect.

For details, refer to the
[Tanh -> Sum -> ScaleShift](@ref dev_guide_attributes_post_ops_with_scales)
example.

That has an implication on the scaling factors passed to the library, however.
Consider the following example of a convolution with \f$\tanh\f$ post-op:

\f[
    \dst_{s8}[:] =
        \frac{1}{scale_{\dst}}
        \cdot
        \tanh(
                scale_{\src}
                \cdot
                scale_{\weights}
                \cdot
                conv_{s32}(\src_{s8}, wei_{s8})
        )
\f]

As you can see:
- The convolution output scales are now
  \f$conv\_output\_scale = scale_{\src} \cdot scale_{\weights}\f$,
  i.e. there is no division by \f$scale_{\dst}\f$;
- And the post-ops scale for \f$\tanh\f$ is set to
  \f$scale\_tanh\_post\_op = \frac{1}{scale_{\dst}}\f$.
