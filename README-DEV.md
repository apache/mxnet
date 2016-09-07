# Workflow for making a release

1. Update `NEWS.md` to list important changes
2. Check out the `stable` branch, merge with `master`.
3. Update `libmxnet_curr_ver` in `deps/build.jl` to the latest commit SHA (or any proper reference). Using `master` here is not good because future changes in libmxnet might break existing Julia packages.
4. Run tests.
5. Commit changes and push.
6. Run `Pkg.tag("MXNet")` in Julia.
7. Run `Pkg.publish()`, which will open a browser for making a pull request to METADATA.jl.
8. Edit the [releases page](https://github.com/dmlc/MXNet.jl/releases) to copy the release notes from `NEWS.md` to the newly created release tag.
