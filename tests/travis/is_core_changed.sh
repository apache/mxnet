#!/bin/bash

# this is a util script to test whether the "core" of
# mxnet has changed. Please modify the regex patterns here
# to ensure the components are covered if you add new "core"
# components to mxnet

# temporarily disable this b/c the OS X tests are failing mysteriously
exit 0

# DEBUG
echo "Files changed in this PR includes:"
echo "**********************************"
git diff --name-only HEAD^
echo "**********************************"

# we ignore examples, and docs
core_patterns=(
  '^dmlc-core'
  '^matlab'
  '^plugin'
  '^python'
  '^src'
  '^tools'
  '^R-package'
  '^amalgamation'
  '^include'
  '^mshadow'
  '^ps-lite'
  '^scala-package'
  '^tests'
)

for pat in ${core_patterns[@]}; do
  if git diff --name-only HEAD^ | grep "$pat"
  then
    exit
  fi
done

echo "I think we are good to skip this travis ci run now"
exit 1 # means nothing has changed
