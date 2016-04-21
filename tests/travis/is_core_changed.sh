#!/bin/bash

# this is a util script to test whether the "core" of
# mxnet has changed. Please modify the regex patterns here
# to ensure the components are covered if you add new "core"
# components to mxnet

# DEBUG
git status
git branch

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

git --no-pager diff --name-only FETCH_HEAD $(git merge-base FETCH_HEAD master) > changed_names.txt

for pat in ${core_patterns[@]}; do
  if grep "$pat" changed_names.txt
  then
    exit
  fi
done

exit 1 # means nothing has changed
