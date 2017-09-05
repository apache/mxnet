# Dockerized multi-architecture build

These docker files will build mxnet for different architectures using cross compilation and produce
runtime binary artifacts.

To compile for all the supported architectures you can run the script
```
$ ./build_all.sh
```

To build a single arch, you can invoke docker directly:

```
$ docker build -f Dockerfile.build.<arch> -t <tag> .
```

By convention all the Dockerfiles produce the build artifacts on /work/build so they can be copied
after.
