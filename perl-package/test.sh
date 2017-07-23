MXNET_HOME=$1
HOME=$2
export LD_LIBRARY_PATH=${MXNET_HOME}/lib
export PERL5LIB=${HOME}/perl5/lib/perl5

cd ${MXNET_HOME}/perl-package/AI-MXNetCAPI/
perl Makefile.PL INSTALL_BASE=${HOME}/perl5
make install || exit -1

cd ${MXNET_HOME}/perl-package/AI-NNVMCAPI/
perl Makefile.PL INSTALL_BASE=${HOME}/perl5
make install || exit -1

cd ${MXNET_HOME}/perl-package/AI-MXNet/
perl Makefile.PL INSTALL_BASE=${HOME}/perl5
make test || exit -1
