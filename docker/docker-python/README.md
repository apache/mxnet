It uses the latest pip binaries to build docker images. 
The latest released mxnet version will be used by the dockerfiles. 
If you want to use another pip binary, feel free to modify the dockerfile before running the build script. 


use as : ./build_python_dockerfile.sh [tag]

The parameter tag inputs the mxnet version which is used to tag the generated docker image. This does not select a different pip binary itself. 


Tests run:
