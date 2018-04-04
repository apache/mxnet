# MXNet Documentation Build Scripts

This folder contains a variety of scripts to generate the MXNet.io website as well as the docs for different versions of MXNet.

## Contents
* [AddPackageLink.py](AddPackageLink.py) - MXNet.io site data massaging; injects pip version numbers in the different versions' install pages
* [AddVersion.py](AddVersion.py) - MXNet.io site data massaging; injects the versions dropdown menu in the navigation bar
* [build_site_tag.sh](build_site_tag.sh) - takes version tags as input and generates static html; calls `build_all_version.sh` and `update_all_version.sh`
* [build_all_version.sh](build_all_version.sh) - takes version tags as input and builds the basic static html for MXNet.io
* [build_doc.sh](build_doc.sh) - used by the CI system to generate MXNet.io; only triggered by new tags; not meant for manual runs or custom outputs
* [Dockerfile](Dockerfile) - has all dependencies needed to build and update MXNet.io's static html
* [update_all_version.sh](update_all_version.sh) - takes the output of `build_all_version.sh` then uses `AddVersion.py` and `AddPackageLink.py` to update the static html

## CI Flow

1. Calls `build_doc.sh`.
2. `VersionedWeb` folder generated with static html of site; old versions are in `VersionedWeb/versions`.
3. `asf-site` branch from the [incubator-mxnet-site](https://github.com/apache/incubator-mxnet-site) project is checked out and contents are deleted.
4. New site content from `VersionedWeb` is copied into `asf-site` branch and then committed with `git`.
5. [MXNet.io](http://mxnet.io) should then show the new content.

## Manual Generation

Use Ubuntu and the setup defined below, or use the Dockerfile provided in this folder to spin up an Ubuntu image with all of the dependencies. Further info on Docker is provided later in this document. For a cloud image, this was tested on [Deep Learning AMI v5](https://aws.amazon.com/marketplace/pp/B077GCH38C?qid=1520359179176).

**Note**: for AMI users or if you already have Conda, you might be stuck with the latest version and the docs build will have a conflict. To fix this, run `/home/ubuntu/anaconda3/bin/pip uninstall sphinx` and follow this with `pip install --user sphinx==1.5.6`.

If you need to build <= v0.12.0, then use a Python 2 environment to avoid errors with `mxdoc.py`. This is a sphinx extension, that was not Python 3 compatible in the old versions. On the Deep Learning AMI, use `source activate mxnet_p27`, and then install the following dependencies.

### Ubuntu 16.04 Dependencies for Docs Generation

```
sudo apt-get update
sudo apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    doxygen \
    pandoc \
    software-properties-common

# You made need to run `/home/ubuntu/anaconda3/bin/pip uninstall sphinx`
# Recommonmark/Sphinx errors: https://github.com/sphinx-doc/sphinx/issues/3800
# Recommonmark should be replaced so Sphinx can be upgraded
# For now we remove other versions of Sphinx and pin it to v1.5.6

pip install --user \
    beautifulsoup4 \
    breathe \
    CommonMark==0.5.4 \
    h5py \
    mock==1.0.1 \
    pypandoc \
    recommonmark==0.4.0 \
    sphinx==1.5.6

# Setup scala
echo "deb https://dl.bintray.com/sbt/debian /" | sudo tee -a /etc/apt/sources.list.d/sbt.list
sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 2EE0EA64E40A89B84B2DF73499E82A75642AC823
sudo apt-get update
sudo apt-get install -y \
  sbt \
  scala

# Cleanup
sudo apt autoremove -y
```

### Script Usage for Manual Generation
The scripts can be run stand-alone or in conjunction, but `build_all_version.sh` should be run first.

### build_all_version.sh
This will checkout each tag provided as an argument and build the docs in the `apache_mxnet` folder. The output is copied to the `VersionedWeb` folder and each version will have a subfolder in `VersionedWeb/versions/`.

Takes one argument:
* **tag list** - space delimited list of Github tags; Example: "1.1.0 1.0.0 master"

**Example Usage**:
`./build_all_version.sh "1.1.0 1.0.0 0.12.1 0.12.0 0.11.0 master"`

### update_all_version.sh
This uses the output of `build_all_version.sh`. If you haven't built the specific tag yet, then you cannot update it.
You can, however, elect to update one or more tags to target the updates you're making.
Takes three arguments:
* **tag list** - space delimited list of Github tags; Example: "1.1.0 1.0.0 master"
* **default tag** - which version should the site default to; Example: 1.0.0
* **root URL** - for the versions dropdown to change to production or dev server; Example: http://mxnet.incubator.apache.org/

Each subfolder in `VersionedWeb/versions` will be processed with `AddVersion.py` and `AddPackageLink.py` to update the static html. Finally, the tag specified as the default tag, will be copied to the root of `VersionedWeb` to serve as MXNet.io's home (default) website. The other versions are accessible via the versions dropdown menu in the top level navigation of the website.

**Example Usage**:
`./update_all_version.sh "1.1.0 1.0.0 0.12.1 0.12.0 0.11.0 master" 1.1.0 http://mxnet.incubator.apache.org/`

### build_site_tag.sh
This one is useful for Docker, or to easily chain the two prior scripts. When you run the image you can call this script as a command a pass the tags, default tag, and root url.

Takes the same three arguments that update_all_version.sh takes.
It will execute `build_all_version.sh` first, then execute `update_all_version.sh` next.

**Example Usage**:
./build_site_tag.sh "1.1.0 master" 1.0.0 http://mxnet.incubator.apache.org/
Then run a web server on the outputted `VersionedWeb` folder.

## Docker Usage ##

The `Dockerfile` will build all of the docs when you create the docker image. You can also run the scripts listed above to regenerate any tag or collection of tags.

Build like:
sudo docker build -t mxnet:docs-base .

Run like:
sudo docker run -it mxnet:docs-base
