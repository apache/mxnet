# main script of travis
if [ ${TASK} == "lint" ]; then
    python3 dmlc-core/scripts/lint.py mshadow all mshadow mshadow-ps || exit -1
fi

if [ ${TASK} == "doc" ]; then
    doxygen doc/Doxyfile 2>log.txt
    (cat log.txt| grep -v ENABLE_PREPROCESSING |grep -v "unsupported tag" |grep nothing) && exit -1
fi

if [ ${TASK} == "build" ]; then
    cd guide
    echo "USE_BLAS=blas" >> config.mk
    make all || exit -1
    cd mshadow-ps
    echo "USE_BLAS=blas" >> config.mk
    echo "USE_RABIT_PS=0" >> config.mk    
    make local_sum.cpu || exit -1
fi
