## Testing convergence of translated network

`testconvergence.py` tests convergence of translated networks. It does the following things in sequence:
1. Takes a directory as input which contains
   - a set of Caffe training/validation and solver prototxts and 
   - a test description file containing the prototxts to translate and metrics to check for convergence.
2. Translates the given caffe prototxts to MXNet Python files.
3. Run training using the translated MXNet code.
4. Checks if training converged according to the criteria given in the test description file.

### How to run testconvergence.py?

Testconvergence.py is a Python script that can be run as shown below:

```
python3 testconvergence.py <test_dir>
```

where, `test_dir` is a directory that has the following structure:

```
sample_test_dir/
├── CaffeModels
├── data
├── snapshots
└── test.cfg
```

`test.cfg` contains a list of tests to run. Each test is specified in the following format:

```
<train_val_prototxt_path> <solver_prototxt_path>
	<metric_name> <GT|LT> <metric_value>
```

where,
- `train_val_prototxt_path` is the path to the training/validation prototxt relative to `<test_dir>`
- `solver_prototxt_path` is the path of the solver prototxt relative to `<test_dir>`
- `metric_name` is the name of a validation metric
- `GT|LT` is `GT` to check if the observed metric is greater than <metric_value> or `LT` to check if the observed metric is lesser than the `<metric_value>`
	
`data` contains the data required for the training.

`snapshot` is an optional folder where snapshots will be written if the solver prototxt is configured to do so.

Please check the [sample](https://github.com/apache/incubator-mxnet/tree/caffe_translator/tools/caffe_translator/test/sample_test_dir) test directory for an example.

### Where does the test results go?
The test script writes the test report, translated files and training logs to `<test_dir>/output/`
