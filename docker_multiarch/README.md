# Dockerized multi-architecture build

These docker files and utilities will build mxnet and run tests for different architectures using cross compilation and produce
runtime binary artifacts.

This utilities require that you have docker installed. [Docker CE](https://docs.docker.com/engine/installation/linux/docker-ce/ubuntu/#install-docker) is recommended.


To compile for all the supported architectures you can run the script
```
$ ./tool.py
```

To build a single arch, you can invoke docker directly:

```
$ docker build -f Dockerfile.build.<arch> -t <tag> .
```

Or call the dockerfile directly:

```
docker build -f <dockerfile> -t <tag> .
```

Or pass the architecture id to the tool:
```
$ ./tool.py -a ubuntu-17.04
```

By convention all the Dockerfiles produce the build artifacts on /work/build so they can be copied
after.


The tool will leave the resulting artifacts on the build/ directory

# TODO

- Handle dependencies between docker files, for example having a yaml file with the dependency graph
  so they can be built in the right order. Right now the dependency is very simple so simple
  alphabetical sorting of the images does the trick.

