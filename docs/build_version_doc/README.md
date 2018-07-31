# MXNet Documentation Build Scripts

This folder contains a variety of scripts to generate the MXNet.io website as well as the docs for different versions of MXNet.

## Contents of the build_version_doc Folder

* [AddPackageLink.py](AddPackageLink.py) - MXNet.io site data massaging; injects pip version numbers in the different versions' install pages
* [AddVersion.py](AddVersion.py) - MXNet.io site data massaging; injects the versions dropdown menu in the navigation bar
* [build_site_tag.sh](build_site_tag.sh) - takes version tags as input and generates static html; calls `build_all_version.sh` and `update_all_version.sh`
* [build_all_version.sh](build_all_version.sh) - takes version tags as input and builds the basic static html for MXNet.io
* [Dockerfile](Dockerfile) - has all dependencies needed to build and update MXNet.io's static html
* [update_all_version.sh](update_all_version.sh) - takes the output of `build_all_version.sh` then uses `AddVersion.py` and `AddPackageLink.py` to update the static html

## Dependencies

The `make docs` process triggers several different documentation functions. Most of this logic is found in [mxdoc.py](https://github.com/apache/incubator-mxnet/blob/master/docs/mxdoc.py). If you review this file you will see several functions which will generate different API documentation sets. For example, they are called in this way:

```python
app.connect("builder-inited", generate_doxygen)
app.connect("builder-inited", build_scala_docs)
app.connect("builder-inited", build_clojure_docs)
# app.connect("builder-inited", build_r_docs)
```

These four are mentioned here in dependencies since you will need to install several extra packages beyond just Sphinx and its collection of plugins and packages. You can, if you wish, simply comment out any of these lines to skip generating the related docs, thereby eliminating several dependencies, and reducing complexity in your builds.

Note: If you review the file, you will see that the function to generate the R docs is commented out by default. Remove this comment mark to generate R docs manually.

Covering each one, here are the related dependencies.

* core MXNet dependencies - you need to build MXNet from source before generating the docs. Once you have accomplished this, you have further dependencies to consider for docs.

* `generate_doxygen` - this generates the Python and C++ APIs.

To enable this manually, run the following (on Ubuntu). Similar packages can be found for other operating systems, but pay attention to versions. Newer versions have been found to have incompatibilities between packages.

```bash
apt-get install \
    doxygen \
    pandoc

pip install --upgrade pip && pip install \
    beautifulsoup4 \
    breathe \
    CommonMark==0.5.4 \
    h5py \
    mock==1.0.1 \
    pypandoc \
    recommonmark==0.4.0 \
    sphinx==1.5.6
```

* `build_scala_docs` - this generates the MXNet-Scala package & Scala API docs.

To enable this manually, run the following:

```bash
apt-get install software-properties-common
apt-get update
apt-get install openjdk-8-jdk
apt-get install openjdk-8-jre

echo "deb https://dl.bintray.com/sbt/debian /" | tee -a /etc/apt/sources.list.d/sbt.list
apt-key add sbt.gpg
apt-get update && apt-get install \
    maven \
    sbt \
    scala
```

* `build_clojure_docs` - this generates the Clojure API docs.

To enable this manually, run the following:

```bash
wget https://raw.githubusercontent.com/technomancy/leiningen/stable/bin/lein
chmod 775 lein
sudo cp lein /usr/local/bin
```

* `build_r_docs` - this generates the R API docs in a PDF file found at `/docs/api/r/mxnet-r-reference-manual.pdf`.

To enable this manually, run the following, and be sure to re-enable the `build_r_docs` function in `mxdoc.py` before you run `make docs`:

```bash
apt-get install \
    libcairo2-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libxt-dev \
    r-base \
    r-base-dev

Rscript -e "install.packages('devtools', repo = 'https://cran.rstudio.com')"
cd R-package
Rscript -e "library(devtools); library(methods); options(repos=c(CRAN='https://cran.rstudio.com')); install_deps(dependencies = TRUE)"
```


## Setting Up a Docs Dev Server

Running docs builds locally on a Mac is not recommended. See [#10858](https://github.com/apache/incubator-mxnet/issues/10858) for workarounds.

For these instructions, you will use an Ubuntu machine. This flow has been tested on a vanilla Ubuntu 16.04 cloud instance on AWS.

**Step 1:** Spin up your Ubuntu server and SSH in.

**Step 2:** Create a Python 2.7 virtual environment (see note).

```bash
sudo apt install virtualenv
virtualenv -p python2.7 mxnet_docs
source mxnet_docs/bin/activate
```

**Note:** Using a Python 2.7 environment is required to build older versions of the docs that have Python 3 incompatibilities. If you're only building the latest or version 1.0.0+, then you may use a Python 3 environment.

**Step 3:** Clone the repo or your own fork of the repo.

```bash
git clone --recursive https://github.com/apache/incubator-mxnet.git mxnet
cd mxnet/docs/build_version_doc
```

**Step 4:** Install dependencies and build MXNet.

This script will install the dependencies for you and build MXNet from source.

```bash
./setup_docs_ubuntu.sh
```

**Step 5a:** Make the docs.

Here you have two options (recommended for most situations):

* Change branches first, or build master with the following:

```bash
cd mxnet
make docs USE_OPENMP=1
```

**Step 5b:** Alternative option for all versions to be built.

This will have the versions dropdown and any post-build processing that generates site artifacts and other requirements for the production site.

**NOTE:** This option only builds what is checked into each branch.

The following script will build all of the latest versions, set the default site to be `master` and use your dev server's IP or DNS for navigation items.

```bash
./build_site_tag.sh '1.2.0;1.1.0;1.0.0;0.12.0;0.11.0;master' master http://your-ip-or-dns
```

**Final Step:** Serve and test.

Refer to [Serving Your Development Version](#serving-your-development-version) for detailed instructions.


**Troubleshooting**: for users that have previously installed Python dependencies, or if you already have Conda, you might be stuck with the latest version of Sphinx and the docs build will have a conflict. To fix this, run `/home/ubuntu/anaconda3/bin/pip uninstall sphinx` and follow this with `pip install --user sphinx==1.5.6`.

If you need to build <= v0.12.0, then use a Python 2 environment to avoid errors with `mxdoc.py`. This is a sphinx extension, that was not Python 3 compatible in the old versions. On the Deep Learning AMI, use `source activate mxnet_p27`, and then install the following dependencies.


## Full Website Build
The following three scripts will help you build multiple version tags and deploy a full site build that with each available API version. If you just want to run master or your current fork's branch you should skip ahead to the [Developer Instructions](#developer-instructions).

The full site build scripts can be run stand-alone or in conjunction, but `build_all_version.sh` should be run first.

### build_all_version.sh
This will checkout each tag provided as an argument and build the docs in the `apache_mxnet` folder. The output is copied to the `VersionedWeb` folder and each version will have a subfolder in `VersionedWeb/versions/`.

Takes one argument:
* **tag list** - semicolon delimited list of Github tags; Example: "1.1.0;1.0.0;master"

**Example Usage**:
`./build_all_version.sh "1.1.0;1.0.0;0.12.1;0.12.0;0.11.0;master"`

### update_all_version.sh
This uses the output of `build_all_version.sh`. If you haven't built the specific tag yet, then you cannot update it.
You can, however, elect to update one or more tags to target the updates you're making.
Takes three arguments:
* **tag list** - space delimited list of Github tags; Example: "1.1.0 1.0.0 master"
* **default tag** - which version should the site default to; Example: 1.0.0
* **root URL** - for the versions dropdown to change to production or dev server; Example: http://mxnet.incubator.apache.org/

Each subfolder in `VersionedWeb/versions` will be processed with `AddVersion.py` and `AddPackageLink.py` to update the static html. Finally, the tag specified as the default tag, will be copied to the root of `VersionedWeb` to serve as MXNet.io's home (default) website. The other versions are accessible via the versions dropdown menu in the top level navigation of the website.

**Example Usage**:
`./update_all_version.sh "1.1.0;1.0.0;0.12.1;0.12.0;0.11.0;master" 1.1.0 http://mxnet.incubator.apache.org/`

### build_site_tag.sh
This one is useful for Docker, or to easily chain the two prior scripts. When you run the image you can call this script as a command a pass the tags, default tag, and root url.

Takes the same three arguments that update_all_version.sh takes.
It will execute `build_all_version.sh` first, then execute `update_all_version.sh` next.

**Example Usage**:
./build_site_tag.sh "1.1.0;master" 1.1.0 http://mxnet.incubator.apache.org/
Then run a web server on the outputted `VersionedWeb` folder.


## Developer Instructions

### Build Docs for Your Current Branch
From the MXNet source root run:

```bash
make docs USE_OPENMP=1
```

If you set `MXNET_DOCS_BUILD_MXNET=0`, the doc generation will skip the build step. This is useful when you have already built MXNet locally.

The files from `make docs` are viewable in `docs/_build/html/`.

**NOTE:** `make docs` doesn't add any version information, and the versions dropdown in the top level navigation is not present. UI bugs can be introduced when the versions dropdown is included, so just testing with `make docs` may be insufficient.

**IMPORTANT:** There are several post-build modifications to the website. This may be responsible for magical or unexplained site behavior. Refer to [Full Site Build Instructions](#full-website-build) and the [Developer Notes](#developer-notes) for more information.


### Developer Notes

1. `AddVersion.py` depends on Beautiful library, which requires target html files to have close tags. Although open tag html can still be rendered by browser, it will be problematic for Beautifulsoup. **This is why the install/index.md page has many rendering issues and is very brittle.**

2. `AddVersion.py` and `AddPackageLink.py` manipulates content for website. If there are layout changes, it may break these two scripts. You will need to change scripts respectively. The [Full Site Build Instructions](#full-website-build) and related scripts leverage these files. If you want to add further post-build site manipulations, this is your starting point, but make sure you include these processes in the related site build scripts.

3. A table of contents feature is used on many pages. The docs build looks for toc-tree tags, so any additions to content may require updates to these tags.

4. The install index page is used for installation validation by triggering from the comment tags, so do not alter these without reviewing the related CI processes.


### Serving Your Development Version

You can view the generated docs with whatever web server you prefer. The Ubuntu setup script described earlier provides instructions for Apache2. MacOS comes preinstalled with Apache.


#### Serve the Website with Apache2

```bash
sudo apt-get install -y apache2
sudo ufw allow 'Apache Full'
```
Copy your `docs/_build/html/` files to where your Apache server will pick them up. If you used the default Ubuntu setup, then this will be `/var/www/html`.

For example, for a simple local development build, from the MXNet root folder:

```bash
cd docs/_build/html
sudo cp -r . /var/www/html/
```

Or if you're using the output from the [Full Website Build](#full-website-build), from the `build_doc_version` folder:

```bash
cd VersionedWeb
sudo cp -r . /var/www/html/
```

**Note**: When generating docs, many files and folders can be deleted or renamed, so it is a good practice to purge the web server directory first, or else you will have old files hanging around potentially introducing errors or hiding broken links.


#### Serve the Website with Python3
Python has a simple web server that you can use for a quick check on your site build. If your SSH tunnel breaks, the site will stop working, so if you plan to share your work as preview in a PR, use Apache2 instead.

From the MXNet source root run:

```bash
cd docs/_build/html
python3 -m http.server
```


### Enabling Redirects
The website uses redirects with `mod_rewrite` for content and folders that have moved. To simulate this locally you need to configure Apache to allow the rewrite module.

```bash
sudo a2enmod rewrite
```

To enable redirects for the folder with the website you need to edit `/etc/apache2/apache2.conf`.
Change directives for `Directory` for `/var/www/html`, or wherever the build files reside, using the following:

```
AllowOverride all
```

Then restart Apache on Ubuntu with:

```bash
sudo systemctl restart apache2
```


## Docker Usage

The `Dockerfile` will build all of the docs when you create the docker image. You can also run the scripts listed above to regenerate any tag or collection of tags.

Build like:
sudo docker build -t mxnet:docs-base .

Run like:
sudo docker run -it mxnet:docs-base


## Deploying the Website to Production

The production website is hosted from the `asf-site` branch of [https://github.com/apache/incubator-mxnet-site](https://github.com/apache/incubator-mxnet-site).

To deploy your website build, you must checkout this repo, delete all of the content, copy your build in, and submit a PR for the update.

There are several manual and semi-automatic processes to be aware of, but the bottom line is that your build should have all of the necessary artifacts for the proper functioning of the site:

1. The root should have the current `.htaccess` file from master in `/docs/`. Make sure you've updated this in master and included the most recent version in your PR.
2. The css file from master `/docs/_static/` will be needed. Be sure that the different versions of the site work. They might need the old version, but the newer version might fix bugs that were in the tags from the legacy versions.
3. Pay attention to `mxdocs.py` as some docs modifications are happening there.
