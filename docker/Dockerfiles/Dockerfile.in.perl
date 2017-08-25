# -*- mode: dockerfile -*-
# part of the dockerfile to install the perl binding

COPY install/perl.sh install/
RUN install/perl.sh && \
    cd /mxnet/perl-package/AI-MXNetCAPI/ && perl Makefile.PL && make install && \
    cd /mxnet/perl-package/AI-NNVMCAPI/ && perl Makefile.PL && make install && \
    cd /mxnet/perl-package/AI-MXNet/ && perl Makefile.PL && make install
