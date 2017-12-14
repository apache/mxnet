# MXNet Profiler Examples

This folder contains examples of using MXNet profiler to generate profiling results in json files.
Please refer to [this link](http://mxnet.incubator.apache.org/faq/perf.html?highlight=profiler#profiler)
for visualizing profiling results and make sure that you have installed a version of MXNet compiled
with `USE_PROFILER=1`.

- profiler_executor.py. To run this example, simply type `python profiler_executor.py` in terminal.
It will generate a json file named `profile_executor_5iter.json`.

- profiler_imageiter.py. You first need to create a file named `test.rec`,
which is an image dataset file before running this example.
Please follow
[this tutorial](https://mxnet.incubator.apache.org/faq/recordio.html?highlight=rec%20file#create-a-dataset-using-recordio)
on how to create `.rec` files using an existing tool in MXNet. After you created 'test.rec',
type `python profiler_imageiter.py` in terminal. It will generate `profile_imageiter.json`.

- profiler_matmul.py. This example profiles matrix multiplications on GPU. Please make sure
that you have installed a GPU enabled version of MXNet before running this example. Type
`python profiler_matmul.py` and it will generate `profile_matmul_20iter.json`.

- profiler_ndarray.py. This examples profiles a series of `NDArray` operations. Simply type
`python profiler_ndarray.py` in terminal and it will generate `profile_ndarray.json`.