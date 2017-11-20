#!/bin/bash
set -e

if [[ -a .git/shallow ]]; then git fetch --unshallow; fi
julia -e 'Pkg.clone(pwd())'
(
    cd `julia -e 'println(Pkg.dir("MXNet", "deps"))'` &&
    ln -fs $TRAVIS_BUILD_DIR/deps/src
)
julia -e 'Pkg.build("MXNet"); Pkg.test("MXNet"; coverage=true)'
