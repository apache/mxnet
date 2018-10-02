#!/bin/sh

# be invoked from build.jl

set -e

diff ../../cblas.h include/cblas.h || cp -v ../../cblas.h include/cblas.h
