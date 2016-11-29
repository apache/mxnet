#!/bin/bash

# Removing this so a previous index.md doesn't get appended to.
rm -f index.md

echo -e "# MXNet - R API HTML Documentation" >> index.md

for f in `ls | grep .*html | grep -v 'mxnet\.html'`
do
    echo -e "## $f\n" | sed 's/\.html//' >> index.md
    html2text $f | tail -n+6 | grep -v "* * *" | grep -v "Package _mxnet_" >> index.md
done
