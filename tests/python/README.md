Python Test Case
================
This folder contains test cases for mxnet in python.

* [common](common) contains common utils for all test modules.
  - From subfolders, import with ```from ..common import get_data```
* [unittest](unittest) contains unit test component for each modules.
  - These are basic tests that must pass for every commit.
* [train](train) contains tests that runs on real network training.
  - These tests can be time consuming.
