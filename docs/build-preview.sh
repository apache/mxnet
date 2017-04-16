#!/bin/bash

# Script to build the HTML docs and serve them.
# Run within docker container for best results.

echo "Building MXNet documentation..."
make clean
make html
echo "Done building MXNet documentation..."

echo "Serving MXNet docs on port 8008..."
cd _build/html
python -m SimpleHTTPServer 8008 

