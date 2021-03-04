Data Types {#dev_guide_data_types}
==================================

oneDNN functionality supports a number of numerical
data types. IEEE single precision floating point (fp32) is considered
to be the golden standard in deep learning applications and is supported
in all the library functions. The purpose of low precision data types
support is to improve performance of compute intensive operations, such as
convolutions, inner product, and recurrent neural network cells
in comparison to fp32.

| Data type | Description
| :---      | :---
| f32       | [IEEE single precision floating point](https://en.wikipedia.org/wiki/Single-precision_floating-point_format#IEEE_754_single-precision_binary_floating-point_format:_binary32)
| bf16      | [non-IEEE 16-bit floating point](https://software.intel.com/content/www/us/en/develop/download/bfloat16-hardware-numerics-definition.html)
| f16       | [IEEE half precision floating point](https://en.wikipedia.org/wiki/Half-precision_floating-point_format#IEEE_754_half-precision_binary_floating-point_format:_binary16)
| s8/u8     | signed/unsigned 8-bit integer

## Inference and Training

oneDNN supports training and inference with the following data types:

| Usage mode | CPU                | GPU                     |
| :---       | :---               | :---                    |
| Inference  | f32, bf16, s8/u8   | f32, bf16, f16, s8/u8   |
| Training   | f32, bf16          | f32, bf16               |

@note
    Using lower precision arithmetic may require changes in the deep learning
    model implementation.

See topics for the corresponding data types details:
 * @ref dev_guide_inference_int8
   * @ref dev_guide_attributes_quantization
 * @ref dev_guide_training_bf16

Individual primitives may have additional limitations with respect to data type
support based on the precision requirements. The list of data types supported
by each primitive is included in the corresponding sections of the developer
guide.

## Hardware Limitations

While all the platforms oneDNN supports have hardware acceleration for
fp32 arithmetics, that is not the case for other data types. Support for low
precision data types may not be available for older platforms. The next sections
explain limitations that exist for low precision data types for
Intel(R) Architecture processors, Intel Processor Graphics and
Xe architecture-based Graphics.

### Intel(R) Architecture Processors

oneDNN performance optimizations for Intel Architecture Processors are
specialized based on Instruction Set Architecture (ISA).
The following ISA have specialized optimizations in the library:
* Intel Streaming SIMD Extensions 4.1 (Intel SSE4.1)
* Intel Advanced Vector Extensions (Intel AVX)
* Intel Advanced Vector Extensions 2 (Intel AVX2)
* Intel Advanced Vector Extensions 512 (Intel AVX-512)
* Intel Deep Learning Boost (Intel DL Boost)

The following table indicates the minimal supported ISA for each of the data
types that oneDNN recognizes.
| Data type | Minimal supported ISA
| :---      | :---
| f32       | Intel SSE4.1
| s8, u8    | Intel AVX2
| bf16      | Intel DL Boost with bfloat16 support
| f16       | not supported

@note
  See @ref dev_guide_int8_computations in the Developer Guide for additional
  limitations related to int8 arithmetic.

@note
  The library has functional bfloat16 support on processors with
  Intel AVX-512 Byte and Word Instructions (AVX512BW) support for validation
  purposes. The performance of bfloat16 primitives on platforms without
  hardware acceleration for bfloat16 is 3-4x lower in comparison to
  the same operations on the fp32 data type.

### Intel(R) Processor Graphics and Xe architecture-based Graphics

Intel Processor Graphics provides hardware acceleration for fp32 and fp16
arithmetic. Xe architecture-based Graphics additionally provides
acceleration for int8 arithmetic (both signed and unsigned). Implementations
for the bf16 data type are functional only and do not currently provide
performance benefits.

| Data type | Support level
| :---      | :---
| f32       | optimized
| bf16      | functional only
| f16       | optimized
| s8, u8    | optimized for Xe architecture-based Graphics (code named DG1 and Tiger Lake)
