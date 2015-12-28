#!/bin/bash
file=scala_test_results.txt

testFail=$(grep -ci "All tests passed" $file)
if [ "$testFail" == "0" ]; then
  cat $file
  echo "Some unit tests failed. "
  exit 1
else
  echo "All unit tests passed! "
fi
