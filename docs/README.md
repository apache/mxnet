# MXNet documentation

A built version of document is available at http://mxnet.dmlc.ml

## To build docs with Docker

The `Dockerfile` in this directory encapsulates all the dependencies needed
to build the docs.  The default entrypoint builds the docs and serves them
through a simple HTTP server for previewing.

```
docker build -t mxnet/docs .
docker run -it -p 8008:8008 mxnet/docs
open http://localhost:8008/
```

### Faster iterative development

If you are working on the docs and want to rebuild them without creating a new
docker image each time, you can do this with

```
docker run -it -p 8008:8008 -v `pwd`:/opt/mxnet/docs mxnet/docs
```

which maps your current directory into the docker image to get any local 
changes.

**NOTE:** Any changes to the API reference will not get rebuilt this way.
The API reference docs are introspected from the built binaries, which 
in this Dockerfile are pulled from github/dmlc/master.  To work-around
this, map a volume with your code changes into the container, and rebuild
MXNet in the container before doing the doc build.  Or use the local
build described below.

## Local build

To build the documentation without docker on your local machine, first
install the required packages for Ubutun 14.04.  These are approximately:

```
sudo apt-get install doxygen python-pip
sudo pip install sphinx==1.3.5 CommonMark==0.5.4 breathe mock==1.0.1 recommonmark
```

(Refer to the Dockerfile for a more reliable description of the dependencies.)
Once the MXNet binaries are built, and you have the dependencies installed,
you can build the docs with:

```make html```
