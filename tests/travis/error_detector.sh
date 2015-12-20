#!/bin/bash
file=scala_test_results.txt

testFail=$(grep -ci "[ERROR]" $file)
if [ "$testFail" != "0" ]; then
  echo "Some unit tests failed. "
  exit 1
fi