# General Notes and Details

## Return status

Returns `1` if any submitted tests returned status `FAILED` or `UNIMPLEMENTED`,
`0` otherwise.

## Running Tests

oneDNN comes with its own testing infrastructure enabled through CMake. Tests
can be executed via the command:
``` sh
    make test_<test-name>
```
This instructs CMake to build a deployable project and run the specific test.

These tests target specific oneDNN features and are based on benchdnn
configurable executions.

The available tests can be found in the oneDNN directory:
tests/benchdnn/inputs/<primitive_name>/test_<test-name>.

## Glossary

| Abbreviation | Description
| :---         | :---
| src          | Source/input image
| wei          | Weights (or filter)
| bia          | Bias
| dst          | Destination/output image
| acc          | Accumulation (typically in terms of data type)

## Modes

**benchdnn** supports several execution flows ("modes"):
* Correctness mode: In this flow the driver performs a correctness validation of
  the library functionality by calling the library API, filling the input data
  according to a certain strategy solely defined by the driver, executing the
  library call and a reference path available in the driver. Then it compares
  the output of both, per element or based on norm, depending on the problem
  setup. In case of results mismatch, the driver will usually report a point
  order, its position in tensor, a reference float value, a reference value
  casted to input data type, a library value, absolute and relative errors. If
  norm check was performed, it reports norm values of reference and library
  results. In addition to result validation, the driver checks that padded area,
  if present in destination memories, remained filling with zeros. In case of
  results mismatch, the driver will report a point order, argument affected,
  expected value (which is always zero) and value got.
* Performance mode: In this flow the driver collects and reports the performance
  statistics of given problems. To collect performance numbers, the driver uses
  a time criterion - runs a problem several rounds accumulating the execution
  time of each round until the sum exceeds the limit border. Once the limit is
  reached, reports numbers and processes the next problem. The limit is
  controlled by a `--max-ms-per-prb=N` option. Instead of time, the number of
  rounds can be set as a criterion, which is controlled by a
  `--fix-times-per-prb=N` option. Refer to
  [performance options](knobs_common.md) for details.
* Correctness & Performance mode: This is a combination of two modes running
  consecutively, first correctness, then performance.
* Listing mode: In this flow the driver constructs the problem, prints its
  reproducer line, and then moves to the next problem. It is also known as a
  dry run. This mode is useful to extract the full list of problems from an
  input file.

## Problem Statuses

Each problem in **benchdnn** has its status indicating the result of running a
problem in the correctness mode. Following statuses are supported:
* `PASSED`. It means that a problem passed the validation, and a library output
  coincides with a reference path from the driver.
* `SKIPPED`. It means that a problem was not run and a brief reason is reported.
* `LISTED`. It means that a benchdnn problem was created and the reproducer line
  was reported. A primitive descriptor is not created in this case.
* `MISTRUSTED`. It means that correctness validation is invalid. This often
  happens when the result has more zeros than the threshold set for the number
  of zero values in the output. One possible reason is incorrect filling with
  input data for a given problem. Treated as `PASSED`.
* `FAILED`. It means that a problem did not pass the validation, and a library
  output differs from a reference path from the driver.
* `UNIMPLEMENTED`. It means that the library does not have an implementation for
  a requested problem. It is treated as `FAILED`.

## Input Files Naming Convention

Benchdnn follows certain [guidelines](benchdnn_input_files_naming_convention.md)
regarding input files naming convention.
