---
layout: page_category
title: Using runtime compilation (RTC) to write CUDA kernels in MXNet
category: faq
faq_c: Extend and Contribute to MXNet
question: How do I implement GPU functions in MXNet using RTC?
permalink: /api/faq/using_rtc
---
<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# Using runtime compilation (RTC) to write CUDA kernels in MXNet

## Introduction

CUDA kernel is a function running on the GPU to perform computation. This tutorial assumes the
reader has a basic knowledge about how to write such kernels.

There are currently 2 typical ways of writing and launching CUDA kernels in MXNet. The first one is
to use the `Kernel<...>::Launch()` API, which is suitable for simple elementwise operations and
enables writing only portion of the kernel, leaving the launch mechanism to MXNet. The
other one is to write a kernel from scratch and launch it using the `<<<...>>>` method from CUDA.
Starting from MXNet 2.0, there is a third option - runtime compilation (RTC). This differs from the
previous methods (which use kernels compiled ahead of time), as it compiles the needed kernels
during runtime of the user script.

In this tutorial we will cover the reasons for using RTC instead of the other methods, show how to
do it, as well as tips on what to keep in mind when doing it.

## Why RTC?

### Problems with kernels compiled ahead of time

The use of kernels compiled ahead of time in MXNet leads to a few problems, which unfortunately
are mostly invisible in any single PR, but grow over the course of many contributions and result in
serious issues.

In order to understand them, let us look at the typical way kernels are launched in MXNet. This
example shows a launch of the simple kernel, taking a single input of type `DType` and producing
single output of type `OType`:

```cpp
MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, DType, {
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, OType, {
    Kernel<...>::Launch(s, inputs[0].dptr<DType>(), outputs[0].dptr<OType>());
  });
});
```

This launch mechanism uses the `MSHADOW_TYPE_SWITCH` macro, which produces a version of the kernel
for every possible type. In the case of nested usage (as is the case in the example shown) it
produces a version of the kernel for every combination of types. This results in a large number of
kernels being generated.

Another factor that multiplies the number of kernels is that different GPU architectures require
different compiled binaries. Therefore for MXNet to support all of them with a single binary, that
binary needs to contain copies of those kernels for each architecure.

This proliferation of CUDA kernels in the binary leads to multiple issues. The first problem is the
size of the MXNet library - each compiled version of the kernel takes some space in the binary,
which is small but multiplied by the number of all versions (which could reach thousands per
GPU architecture) and GPU architectures. This increase in size led to multiple issues reported with
distribution of the MXNet package,
[building the library](https://github.com/apache/incubator-mxnet/issues/17045) as well as
[limiting the number of architectures natively
supported](https://github.com/apache/incubator-mxnet/pull/18205).

The second issue is the "idle" memory consumption of the MXNet library. In order to efficiently
launch kernels when they are called, CUDA driver needs to transfer them to the GPU memory ahead of
time. Since it cannot anticipate which kernels will actually be used, all of the kernels are
transferred when the CUDA context is created on a GPU. This means that, even if a user never uses
e.g. kernel which adds `int8` and `float16` tensors, that kernel still occupies memory on their GPU,
reducing the amount of memory available for useful work.

The third issue, mostly affecting MXNet developers, is the compilation time of the MXNet library.
The more kernels versions need to be compiled, the more time and hardware resources is needed.

### RTC to the rescue!

All of the issues mentioned in the previous paragraph are solved when using runtime compilation.
Using this paradigm, only the kernels actually invoked in the user script are compiled. They do not
occupy space in the MXNet binary and there is no unused kernels stored in users' GPU memory.

RTC also enables more features:

 - using more information about specific usage of the kernel when compiling it (e.g. using shape
   information of the inputs) to optimize it better
 - writing kernels accepting any combinations of input and output types
 - (in the future) fusing more operations into the generated kernels.

## RTC for kernel developers

### Example: unary operators

Let us start with an example of the simple kernel written using RTC: a kernel which performs unary
operation (with a concrete example of sigmoid) on its input. It is not a toy example though: it is
a fully generic kernel, capable of operating on any combination of input and output types, as well
as applying any unary operator:

```cpp
struct UnaryRTCCompute {
  std::string OP;

  void operator()(const nnvm::NodeAttrs& attrs,
                  const OpContext& ctx,
                  const std::vector<TBlob>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<TBlob>& outputs);
};

const char unary_kernel_fwd[] = R"code(

__launch_bounds__(kRTCMaxThreadsPerBlock)
__global__ void unary_kernel(const InputType* input,
                             const OutputType* output,
                             const index_t N) {
  using IType = AccType<InputType>;
  using OType = AccType<OutputType>;

  for (index_t tid = blockIdx.x * blockDim.x + threadIdx.x;
       tid < N;
       tid += gridDim.x * blockDim.x) {
    const auto input = IType::from(input[i]);
    const auto temp = OP(input);  // enables returning different type

    if (req == OpReqType::kAddTo) {
      // temp2 may have a wider type than either temp
      // or OType
      const auto temp2 = op::add(temp, OType::from(output[i]));
      output[i] = OType::to(temp2);
    } else {
      output[i] = OType::to(temp);
    }
  }
}

)code";

void UnaryRTCCompute::operator()(const nnvm::NodeAttrs& attrs,
                                 const OpContext& ctx,
                                 const std::vector<TBlob>& inputs,
                                 const std::vector<OpReqType>& req,
                                 const std::vector<TBlob>& outputs) {
  using namespace mxnet::common::cuda::rtc;
  if (req[0] == kNullOp) return;
  mshadow::Stream<gpu>* s = ctx.get_stream<gpu>();
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);

  const std::string code = std::string("const OpReqType req = ") +
                           util::to_string(req[0]) +
                           ";\n"
                           "#define OP op::" +
                           OP +
                           "\n"
                           "using InputType = " +
                           common::mshadow_type_info(inputs[0].type_flag_).name +
                           ";\n"
                           "using OutputType = " +
                           common::mshadow_type_info(outputs[0].type_flag_).name +
                           ";\n";

  std::vector<const void*> args;
  const index_t size = outputs[0].Size();
  args.emplace_back(&(inputs[0].dptr_));
  args.emplace_back(&(outputs[0].dptr_));
  args.emplace_back(&size);

  auto kernel = get_function(code, "unary_kernel", unary_kernel_fwd,
                             ctx.run_ctx.get_ctx().dev_id);

  const int n_threads = 512;
  const index_t n_blocks = (size + n_threads - 1) / n_threads;
  const int shared_memory_size = 0;
  launch(kernel, {n_blocks, 1, 1}, {512, 1, 1},
         shared_memory_size, s, &args);
}

NNVM_REGISTER_OP(sigmoid)
.set_attr<FCompute>("FCompute<gpu>", UnaryRTCCompute{"sigmoid"});
```

### Kernels are text...

The main difference when writing kernels using RTC is that the kernel code becomes the text string.
This means that it is possible to change or compose the code at runtime, as is done here:

```cpp
  const std::string code = std::string("const OpReqType req = ") +
                           util::to_string(req[0]) +
                           ";\n"
                           "#define OP op::" +
                           OP +
                           "\n"
                           "using InputType = " +
                           common::mshadow_type_info(inputs[0].type_flag_).name +
                           ";\n"
                           "using OutputType = " +
                           common::mshadow_type_info(outputs[0].type_flag_).name +
                           ";\n";
```

where the operation `OP` is also provided as a string in the operator declaration:

```cpp
NNVM_REGISTER_OP(sigmoid)
.set_attr<FCompute>("FCompute<gpu>", UnaryRTCCompute{"sigmoid"});
```

### and do not know MXNet source code

How does the kernel know what operation it should perform? The kernel's source code uses `OP`,
which shows up in the `code` variable and is equal to `op::sigmoid`. Let us compare this to how the
same operator is defined for CPU:

```cpp
MXNET_OPERATOR_REGISTER_UNARY(sigmoid)
.set_attr<FCompute>("FCompute<cpu>", UnaryOp::Compute<cpu, mshadow_op::sigmoid>)
```

Since the kernel is compiled at runtime, it does not have access to the rest of the MXNet source
code, including `mshadow_op.h`, which defined `mshadow_op::sigmoid`. This means that we need to
provide the kernel with definitions of those functions (again, in text string form). Every
RTC-compiled kernel is prepended with a common header, containing string found in
`src/common/cuda/rtc/` directory. The `src/common/cuda/rtc/forward_functions-inl.h` file contains
the definition of `op::sigmoid`:

```cpp
template <typename DType>
__device__ inline DType sigmoid(const DType val) {
  if (type_util::has_double_or_integral<DType>::value) {
    return 1./(1 + ::exp(-val));
  } else {
    return 1.f/(1 + expf(-val));
  }
}
```

### Handling of data types

MXNet has support for many datatypes. Some of those datatypes, like `float16`, `int8` or `bool` are
useful when storing the results, but in many computations they are too limiting as they can easily
overflow in the intermediate stages. That is why in the example we use `AccType<T>` class - it
provides an accumulation type, that is potentially larger than the storage type - for example,
`AccType<float16>::type` is `float32`. It also provides special loading and storing functions:
`AccType<T>::from()` and `AccType<T>::to()`.

One of the features of RTC-enabled kernels is to be able to accommodate any combination of the
input and output datatypes. Using `auto` as the output type of the intermediate steps helps with,
especially since many binary operators return a mixed type:

```cpp
template <typename DType, typename DType2>
__device__ inline typename type_util::mixed_type<DType, DType2>::type
add(const DType a, const DType2 b) {
  return a + b;
}
```

`mixed_type<T, U>::type` is a type capable of storing value of the operation between 2 types `T` and
`U` - e.g. `mixed_type<float64, float32>::type = float64` and `mixed_type<float32, int32>::type =
float32`.

### Compiling and launching RTC kernels

The kernel code stored in `unary_kernel_fwd` is generic and relies on multiple names to be defined,
like `req`, `OP` or `InputType`. This is handled in the specific operator using the kernel by
defining a set of parameters that will be concatenated to the code during compilation:

```cpp
  const std::string code = std::string("const OpReqType req = ") +
                           util::to_string(req[0]) +
                           ";\n"
                           "#define OP op::" +
                           OP +
                           "\n"
                           "using InputType = " +
                           common::mshadow_type_info(inputs[0].type_flag_).name +
                           ";\n"
                           "using OutputType = " +
                           common::mshadow_type_info(outputs[0].type_flag_).name +
                           ";\n";
```

In order to compile the kernel, the `mxnet::common::cuda::rtc::get_function` method is used:

```cpp
  auto kernel = get_function(code, "unary_kernel", unary_kernel_fwd,
                             ctx.run_ctx.get_ctx().dev_id);
```

In order to eliminate overheads coming from the compilation, it uses cache of kernels, with a key
being the name of the kernel (`"unary_kernel"` in our case) and the set of parameters (`code` in our
case). If the kernel is already in cache, it is returned, otherwise compilation takes place. If it
fails, the full source code is saved to disk and the MXNet error with the compilation log is
generated.

To launch the kernel, the `mxnet::common::cuda::rtc::launch` method is used:

```cpp
  launch(kernel, {n_blocks, 1, 1}, {512, 1, 1},
         shared_memory_size, s, &args);
```

It takes the kernel object, grid and block dimensions, size of dynamic shared memory, stream and
kernel parameters.

## Other features enabled by RTC

### Vectorization

The actual kernel used for application of unary operator in MXNet looks slightly different compared
to the simple example shown in the previous paragraph. Differences come from using vectorization.
This means, that instead of reading (or writing) 1 element at a time, kernel instead accesses
multiple array elements at once. This is beneficial, especially when dealing with smaller
types like `float16` or `int8`. Accessing those small types one by one is inefficient and does not
saturate the memory bandwidth of the GPU, so using vector accesses improves achieved memory
bandwidth.

```cpp

// excerpt from src/operator/tensor/elemwise_unary_op.h
struct UnaryRTCCompute {
  std::string OP;

  void operator()(const nnvm::NodeAttrs& attrs,
                  const OpContext& ctx,
                  const std::vector<TBlob>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<TBlob>& outputs);
};

// excerpt from src/operator/tensor/elemwise_unary_op.cc
struct unary_kernel_params {
  const void *inputs[1];
  void *outputs[1];
};

const char unary_kernel_fwd[] = R"code(

struct unary_kernel_params {
  const void *inputs[1];
  void *outputs[1];
};

__launch_bounds__(kRTCMaxThreadsPerBlock)
__global__ void unary_kernel(const unary_kernel_params params,
                             const index_t lead_dim,
                             const index_t other_dim,
                             const index_t N,
                             const index_t num_aligned_elements) {
  using namespace vector;
  VectorizedLoader<InputType0, nvec, aligned> loader(
    reinterpret_cast<const InputType0*>(params.inputs[0]), N);
  VectorizedStorer<OutputType0, nvec, aligned> storer(
    reinterpret_cast<OutputType0*>(params.outputs[0]), N);

  using IType = AccType<InputType0>;
  using OType = AccType<OutputType0>;

  const index_t M = num_aligned_elements;

  for (index_t tid = blockIdx.x * blockDim.x + threadIdx.x;
       tid < M;
       tid += gridDim.x * blockDim.x) {
    loader.load(tid, N);
    if (req == OpReqType::kAddTo) {
      storer.load(tid, N);
    }
#pragma unroll
    for (int i = 0; i < nvec; ++i) {
      const auto input = IType::from(loader.separate()[i]);
      const auto temp = OP(input);  // enables returning different type

      if (req == OpReqType::kAddTo) {
        // temp2 may have a wider type than either temp
        // or OType
        const auto temp2 = op::add(temp, OType::from(storer.separate()[i]));
        storer.separate()[i] = OType::to(temp2);
      } else {
        storer.separate()[i] = OType::to(temp);
      }
    }
    storer.store(tid, N);
  }
}

)code";

void UnaryRTCCompute::operator()(const nnvm::NodeAttrs& attrs,
                                 const OpContext& ctx,
                                 const std::vector<TBlob>& inputs,
                                 const std::vector<OpReqType>& req,
                                 const std::vector<TBlob>& outputs) {
  using namespace mxnet::common::cuda::rtc;
  if (req[0] == kNullOp) return;
  mshadow::Stream<gpu>* s = ctx.get_stream<gpu>();
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);

  const std::string code = std::string("const OpReqType req = ") +
                           util::to_string(req[0]) +
                           ";\n"
                           "#define OP op::" +
                           OP +
                           "\n";
  const int nvec = outputs[0].type_flag_ == mshadow::kFloat64 ? 2 : 4;

  const index_t size = outputs[0].Size();
  unary_kernel_params params = { {inputs[0].dptr_},
                                 {outputs[0].dptr_} };

  VectorizedKernelRTCLauncher(code, "unary_kernel",
                              unary_kernel_fwd, nvec,
                              size, 1, s, params,
                              inputs, outputs,
                              ctx.run_ctx.get_ctx().dev_id);
}

// excerpt from src/operator/tensor/elemwise_unary_op_basic.cu
NNVM_REGISTER_OP(sigmoid)
.set_attr<FCompute>("FCompute<gpu>", UnaryRTCCompute{"sigmoid"});
```

RTC implementation in MXNet provides a few useful helper functions and classes, which simplify the
process of writing and launching kernels using vectorization. For accessing the memory using
vectorization, 2 classes are provided, used in this kernel to access input and output array:

```cpp
  VectorizedLoader<InputType0, nvec, aligned> loader(
    reinterpret_cast<const InputType0*>(params.inputs[0]), N);
  VectorizedStorer<OutputType0, nvec, aligned> storer(
    reinterpret_cast<OutputType0*>(params.outputs[0]), N);
```

The `loader` object accesses `params.inputs[0]` pointer to array of N elements having type
`InputType0` (which is the name assigned to the type of the first input by the
`VectorizedKernelRTCLauncher`, which is the helper launcher function). It loads `nvec` elements at
a time and has additional `aligned` option, which is also set by the `VectorizedKernelRTCLauncher`.
Similarly `storer` object is used to write data of type `OutputType0` to `params.outputs[0]`.

The kernel using `VectorizedKernelRTCLauncher` needs to have specific parameters:

```cpp
__global__ void unary_kernel(const unary_kernel_params params,      // kernel-specific parameters
                             const index_t lead_dim,                // lead dimension of the tensor
                             const index_t other_dim,               // size of the other dimensions
                             const index_t N,                       // total number of elements
                             const index_t num_aligned_elements) {  // number of vector elements in
                                                                    // lead dimension
```
