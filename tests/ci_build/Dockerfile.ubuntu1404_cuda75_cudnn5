FROM nvidia/cuda:7.5-cudnn5-devel
MAINTAINER Ly Nguyen <lynguyen@amazon.com>

# OPENCV
RUN apt-mark hold libcudnn5 libcudnn5-dev
RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y libopencv-dev

# BLAS
RUN apt-get install -y libatlas-base-dev

# PYTHON2
RUN apt-get install -y python-setuptools python-pip python-dev unzip gfortran
RUN pip install numpy nose scipy

# PYTHON3
RUN apt-get install -y python3-setuptools python3-pip
RUN pip3 install numpy nose scipy && ln -s -f /usr/local/bin/nosetests-3.4 /usr/local/bin/nosetests3

# TESTDEPS
RUN apt-get install -y libgtest-dev cmake wget
RUN cd /usr/src/gtest && cmake CMakeLists.txt && make && cp *.a /usr/lib
RUN pip install nose cpplint 'pylint==1.4.4' 'astroid==1.3.6'

# MAVEN
RUN apt-get install -y maven default-jdk

# R
RUN apt-get install -y software-properties-common r-base-core libcurl4-openssl-dev libssl-dev libxml2-dev
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E084DAB9
RUN add-apt-repository -y ppa:marutter/rdev
RUN apt-get update && apt-get -y upgrade
RUN DEBIAN_FRONTEND=noninteractive apt-get -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confnew" install r-base r-base-dev
RUN Rscript -e "install.packages('devtools', repo = 'https://cran.rstudio.com')"
