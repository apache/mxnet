#!/bin/bash

# This script is mainly used to catch errors from maven results for Scala
# to avoid false positives/negatives from Travis's interpretation. 

file=scala_test_results.txt

testFail=$(grep -ci "All tests passed" $file)

if [ "$testFail" == "0" ]; then
  # print results if anything fails
  cat $file
  echo "Some unit tests failed. "
  exit 1
else
  echo "All unit tests passed! "
fi
