# Before building this image you would need to build MXNet by executing:
# docker build -f Dockerfile.build.ubuntu-17.04 -t mxnet.build.ubuntu-17.04 .
# if you haven't done it before.

FROM mxnet.build.ubuntu-17.04

ENV DEBIAN_FRONTEND=noninteractive
#ENV BUILD_OPTS "USE_OPENCV=0 USE_BLAS=openblas GTEST_PATH=/usr/src/googletest/googletest"

##################
# R installation
RUN apt-get update
#RUN apt-get remove -y gnupg
#RUN apt-get install -y --reinstall\
#	 gnupg2 dirmngr

RUN apt-get install -y dirmngr libopencv-dev
RUN echo "deb http://cran.rstudio.com/bin/linux/ubuntu zesty/" >> /etc/apt/sources.list
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E084DAB9

RUN apt-get install -y\
	 r-base r-base-core r-recommended r-base-dev libxml2-dev libxt-dev libssl-dev libcurl4-openssl-dev


WORKDIR /work/mxnet
RUN cp R-package/DESCRIPTION .
RUN Rscript -e "install.packages('devtools', repo = 'https://cran.rstudio.com')"
RUN Rscript -e "library(devtools); library(methods); options(repos=c(CRAN='https://cran.rstudio.com')); install_deps(dependencies = TRUE)"


##################
# MXNet R package
RUN make rpkg 
RUN R CMD INSTALL mxnet_current_r.tar.gz
##################

