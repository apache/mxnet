Python Test Case
================
This folder contains test cases for MXNet in python.

* [common](common) contains common utils for all test modules.
  - From subfolders, import with ```from ..common import get_data```
* [unittest](unittest) contains unit test component for each modules.
  - These are basic tests that must pass for every commit.
* [train](train) contains tests that runs on real network training.
  - These tests can be time consuming.

The file 'requirements.test.txt' contains all dependencies need to run unit tests.  It can be
installed with the command 'pip install -r requirements.test.txt'.
