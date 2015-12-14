#!/bin/bash

if [[ -a .git/shallow ]]; then git fetch --unshallow; fi
julia -e 'Pkg.clone(pwd()); Pkg.build("MXNet"); Pkg.test("MXNet"; coverage=true)'
