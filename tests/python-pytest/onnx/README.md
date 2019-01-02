# ONNX tests

## Directory structure:

```bash
.
├── README.md
├── backend.py
├── backend_rep.py
├── backend_test.py
├── gluon_backend_test.py
├── mxnet_backend_test.py
├── mxnet_export_test.py
├── test_cases.py
├── test_models.py
└── test_node.py
```

* `backend.py` - MXNetBackend. This file contains prepare(). \
This class can be used for both, MXNet and Gluon backend.
* `backend_rep.py` - MXNetBackendRep and GluonBackendRep for running inference
* `backend_test.py` - prepare tests by including tests from `test_cases.py`
* `gluon_backend_test.py` - Set backend as gluon and execute ONNX tests for ONNX->Gluon import.
* `mxnet_backend_test.py` - Set backend as gluon and add tests for ONNX->MXNet import/export.
Since MXNetBackend for export, tests both import and export, the test list in this file is
a union of tests that execute for import and export, export alone, and import alone.
* `mxnet_export_test.py` - Execute unit tests for testing MXNet export code - this is not specific to
any operator.
* `test_cases.py` - list of test cases for operators/models that are supported
for "both", import and export, "import" alone, or "export" alone.
* `test_models.py` - custom tests for models
* `test_node.py` - custom tests for operators. These tests are written independent of ONNX tests, in case
ONNX doesn't have tests yet or for MXNet specific operators.