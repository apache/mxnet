#!/bin/bash

if [[ -a .git/shallow ]]; then git fetch --unshallow; fi
julia -e 'Pkg.rm("MXNet")' # in case Jenkins CI did not remove existing files
julia -e 'Pkg.clone(pwd()); Pkg.build("MXNet"); Pkg.test("MXNet"; coverage=true)'
