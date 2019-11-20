--
layout: page_api
title: Multi Threaded Inference
action: Get Started
action_url: /get_started
permalink: /api/cpp/docs/tutorials/multi_threaded_inference
is_tutorial: true
tag: cpp
--
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

## Multi Threaded Inference API

A long standing request from MXNet users has been to invoke parallel inference on a model from multiple threads while sharing the parameters.
With this use case in mind, the threadsafe version of CachedOp was added to provide a way for customers to do multi-threaded inference for MXNet users.
This doc attempts to do the following:
1. Discuss the current state of thread safety in MXNet
2. Explain how one can use C API and thread safe version of cached op, along with CPP package to achieve iultithreaded inference. This will be useful for end users as well as frontend developers of different language bindings
3. Discuss the limitations of the above approach
4. Future Work

## Current state of Thread Safety in MXNet

Examining the current state of thread safety in MXNet we can arrive to the following conclusion:

1. MXNet Dependency Engine is thread safe (except for WaitToRead invoked inside a spawned thread. Please see Limitations section).
2. Graph Executor which is Module/Symbolic/C Predict API backend is not thread safe
3. Cached Op (Gluon Backend) is not thread safe

The CachedOpThreadSafe and corresponding C APIs were added to address point 3 above and provide a way
to do multi-threaded inference.

```
/*!
 * \brief create cached operator, allows to choose thread_safe version
 * of cachedop
 */
MXNET_DLL int MXCreateCachedOpEX(SymbolHandle handle,
                                 int num_flags,
                                 const char** keys,
                                 const char** vals,
                                 CachedOpHandle *out,
                                 bool thread_safe DEFAULT(false));
```

## Multithreaded inference in MXNet with C API and CPP Package

### Prerequisites
To complete this tutorial you need to:
- Learn the basics about [MXNet C++ API](/api/cpp)

### Setup the MXNet C++ API
To use the C++ API in MXNet, you need to build MXNet from source with C++ package. Please follow the [built from source guide](/get_started/ubuntu_setup.html), and [C++ Package documentation](/api/cpp)
The summary of those two documents is that you need to build MXNet from source with `USE_CPP_PACKAGE` flag set to 1. For example: `make -j USE_CPP_PACKAGE=1 USE_CUDA=1 USE_CUDNN=1`.
This example requires a build with CUDA and CUDNN.

### Build the example
If you have built mxnet from source with make, then do the following:

```bash
$ cd example/multi_threaded_inference
$ make
```

If you have built mxnet from source with cmake, please uncomment the specific lines for cmake build or set the following environment variables: `MKLDNN_BUILD_DIR (default is $(MXNET_ROOT)/3rdparty/mkldnn/build)`, `MKLDNN_INCLUDE_DIR (default is $(MXNET_ROOT)/3rdparty/mkldnn/include)`, `MXNET_LIB_DIR (default is $(MXNET_ROOT)/lib)`.

### Download the model and run multi threaded inference example
To download a model use the `get_model.py` script. This downloads a model to run inference.

```python
python3 get_model.py --model <model_name>
```
e.g.
```python
python3 get_model.py --model imagenet1k-inception-bn
```
Only the supported models with `get_model.py` work with multi threaded inference.

To run the multi threaded inference example:

First export `LD_LIBRARY_PATH`:

```bash
$ export LD_LIBRARY_PATH=<MXNET_LIB_DIR>:$LD_LIBRARY_PATH
```

```bash
$ ./multi_threaded_inference [model_name] [num_threads] [is_gpu] [file_names]
```
e.g.

```bash
./multi_threaded_inference imagenet1k-inception-bn 2 1 grace_hopper.jpg dog.jpg
```

The above script spawns 2 threads, shares the same cachedop and params among two threads, and runs inference on GPU. It returns the inference results in the order in which files are provided.

NOTE: This example is to demonstrate the multi-threaded-inference with cached op. The inference results work well only with specific models (e.g. imagenet1k-inception-bn). The results may not necessarily be very accurate because of different preprocessing step required etc.

### Code walkthrough multi-threaded inference with CachedOp

The multi threaded inference example (`multi_threaded_inference.cc`) involves the following steps:

1. Parse arguments and load input image into ndarray
2. Prepare input data and load parameters, copying data to a specific context
3. Preparing arguments to pass to the CachedOp and calling C API to **create cached op**
4. Prepare lambda function which will run in spawned threads. Call C API to **invoke cached op** within the lambda function.
5. Spawn multiple threads and wait for all threads to complete.
6. Post process data to obtain inference results and cleanup.

### Step 1: Parse arguments and load input image into ndarray

```c++
int main(int argc, char *argv[]) {
  if (argc < 5) {
    std::cout << "Please provide a model name, num_threads, is_gpu, test_image" << std::endl
              << "Usage: ./multi_threaded_inference [model_name] [num_threads] [is_gpu] apple.jpg"
              << std::endl
              << "Example: ./.multi_threaded_inference imagenet1k-inception-bn 1 0 apple.jpg"
              << std::endl
              << "NOTE: Thread number ordering will be based on the ordering of file inputs" << std::endl
              << "NOTE: Epoch is assumed to be 0" << std::endl;
    return EXIT_FAILURE;
  }
  std::string model_name = std::string(argv[1]);
  int num_threads = std::atoi(argv[2]);
  bool is_gpu = std::atoi(argv[3]);
  ...
  ...
  mxnet::cpp::Shape input_shape = mxnet::cpp::Shape(1, 3, 224, 224);
  for (size_t i = 0; i < files.size(); i++) {
    files[i].resize(image_size);
    GetImageFile(test_files[i], files[i].data(), channels,
                 cv::Size(width, height));
    input_arrs.emplace_back(mxnet::cpp::NDArray(files[i].data(), input_shape, mxnet::cpp::Context::cpu(0)));
  }
```

The above code parses arguments, loads the image file into a ndarray with a specific shape. There arae few things that are set by default and not configurable. For example, `static_alloc` and `static_shape` are by default set to true.


### Step 2: Prepare input data and load parameters, copying data to a specific context
```c++
void run_inference(const std::string& model_name, const std::vector<mxnet::cpp::NDArray>& input_arrs,
                   std::vector<mxnet::NDArray*> *output_mx_arr,
                   int num_inf_per_thread = 1, bool random_sleep = false,
                   int num_threads = 1, bool static_alloc = false,
                   bool static_shape = false,
                   bool is_gpu = false) {                                                                                       
  ...
  ...
  ...
  // Prepare input data and parameters
  std::vector<mxnet::cpp::NDArray> data_arr(num_threads);
  std::vector<mxnet::cpp::NDArray> softmax_arr;
  std::vector<mxnet::cpp::NDArray> params;
  mxnet::cpp::Shape data_shape = mxnet::cpp::Shape(1, 3, 224, 224);
  mxnet::cpp::Shape softmax_shape = mxnet::cpp::Shape(1);
  int num_inputs = out.ListInputs().size();

  for (size_t i = 0; i < data_arr.size(); ++i) {
    data_arr[i] = input_arrs[i].Copy(ctx);
  }
  prepare_input_data(softmax_shape, ctx, num_threads, &softmax_arr);
  std::map<std::string, mxnet::cpp::NDArray> parameters;
  mxnet::cpp::NDArray::Load(param_file, 0, &parameters);

  for (std::string name : out.ListInputs()) {
    if (name == "arg:data") {
      continue;
    }
    if (parameters.find("arg:" + name) != parameters.end()) {
      params.push_back(parameters["arg:" + name].Copy(ctx));
    } else if (parameters.find("aux:" + name) != parameters.end()) {
      params.push_back(parameters["aux:" + name].Copy(ctx));
    }
  }
```

The above code loads params and copies input data and params to specific context.

### Step 3: Preparing arguments to pass to the CachedOp and calling C API to create cached op

```c++
  CachedOpHandle hdl = CachedOpHandle();

  std::vector<std::string> flag_keys{"data_indices", "param_indices",
                                     "static_alloc", "static_shape"};
  std::string param_indices = "[";
  for (size_t i = 1; i < num_inputs; ++i) {
    param_indices += std::to_string(i);
    param_indices += std::string(", ");
  }
  param_indices += "]";
  std::vector<std::string> flag_vals{"[0]", param_indices, static_alloc_str,
                                     static_shape_str};
  std::vector<const char*> flag_key_cstrs, flag_val_cstrs;
  flag_key_cstrs.reserve(flag_keys.size());
  for (size_t i = 0; i < flag_keys.size(); ++i) {
    flag_key_cstrs.emplace_back(flag_keys[i].c_str());
  }
  for (size_t i = 0; i < flag_vals.size(); ++i) {
    flag_val_cstrs.emplace_back(flag_vals[i].c_str());
  }

  int ret1 = MXCreateCachedOpEX(out.GetHandle(), flag_keys.size(),
                                flag_key_cstrs.data(), flag_val_cstrs.data(),
                                &hdl, true);
  if (ret1 < 0) {
    LOG(FATAL) << MXGetLastError();
  }
```

The above code prepares `flag_key_cstrs` and `flag_val_cstrs` to be passed the Cached op.
The C API call is made with `MXCreateCachedOpEX`. This will lead to creation of thread safe cached
op since the `thread_safe` (which is the last parameter to `MXCreateCachedOpEX`) is set to
true. When this is set to false, it will invoke CachedOp instead of CachedOpThreadSafe.


### Step 4: Prepare lambda function which will run in spawned threads

```c++
  auto func = [&](int num) {
    unsigned next = num;
    if (random_sleep) {
      int sleep_time = rand_r(&next) % 5;
      std::this_thread::sleep_for(std::chrono::seconds(sleep_time));
    }
    int num_output = 0;
    const int *stypes;
    int ret = MXInvokeCachedOpEx(hdl, arr_handles[num].size(), arr_handles[num].data(),
                                 &num_output, &(cached_op_handles[num]), &stypes,
                                 true);
    if (ret < 0) {
      LOG(FATAL) << MXGetLastError();
    }
    mxnet::cpp::NDArray::WaitAll();
    (*output_mx_arr)[num] = static_cast<mxnet::NDArray *>(*cached_op_handles[num]);
  };
```

The above creates the lambda function taking the thread number as the argument.
If `random_sleep` is set it will sleep for a random number (secs) generated between 0 to 5 seconds.
Following this, it invokes `MXInvokeCachedOpEx`(from the hdl it determines whether to invoke cached op threadsafe version or not).
When this is set to false, it will invoke CachedOp instead of CachedOpThreadSafe.

### Step 5: Spawn multiple threads and wait for all threads to complete

```c++
  std::vector<std::thread> worker_threads(num_threads);
  int count = 0;
  for (auto &&i : worker_threads) {
    i = std::thread(func, count);
    count++;
  }

  for (auto &&i : worker_threads) {
    i.join();
  }

  mxnet::cpp::NDArray::WaitAll();
```

Spawns multiple threads, joins and waits to wait for all ops to complete.
The other alternative is to wait in the thread on the output ndarray and remove the WaitAll after join.

### Step 6: Post process data to obtain inference results and cleanup

```c++
  ...
  ...
  for (size_t i = 0; i < num_threads; ++i) {
    PrintOutputResult(static_cast<float *>((*output_mx_arr)[i]->data().dptr_),
                      (*output_mx_arr)[i]->shape().Size(), synset);
  }
  int ret2 = MXFreeCachedOpEX(hdl, true);
  ...
```

The above code outputs results for different threads and cleans up the thread safe cached op.

## Current Limitations

1. Only operators tested with the existing model coverage are supported. Other operators and operator types (stateful operators, custom operators are not supported. Existing model coverage is as follows (this list will keep growing as we test more models with different model types):
|Models Tested|MKLDNN|CUDNN|NO-CUDNN|
| --- | --- | --- | --- |
| imagenet1k-resnet-18 | Yes | Yes | Yes |
| imagenet1k-resnet-152 | Yes | Yes | Yes |
| imagenet1k-resnet-50 | Yes | Yes | Yes |
2. Only dense storage types are supported currently.
3. Multi GPU Inference not supported currently.
4. Instantiating multiple instances of SymbolBlockThreadSafe is not supported. Can run parallel inference only on one model per process.
5. dynamic shapes not supported in thread safe cached op.
6. Bulking of ops is not supported.
7. This only supports inference use cases currently, training use cases are not supported.
8. Graph rewrites with subgraph API currently not supported.
9. Frontend API Changes to support multi threaded inference.
10. Multi threaded inference with threaded engine with Module/Symbolic API and C Predict API are not currently supported.
11. Exception thrown with `wait_to_read` in individual threads can cause issues. Calling invoke from each thread and calling WaitAll after thread joins should still work fine.


## Future Work

Future work includes Increasing model coverage and addressing most of the limitations mentioned under Current Limitations except the training use case.
For more updates, please subscribe to discussion activity on RFC: https://github.com/apache/incubator-mxnet/issues/16431.
