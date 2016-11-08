echo "BUILD make"
cp make/config.mk .
echo "USE_CUDA=1" >> config.mk
echo "USE_CUDA_PATH=/usr/local/cuda" >> config.mk
echo "USE_CUDNN=1" >> config.mk
make -j 4 || exit -1

echo "BUILD lint"
make lint || exit -1

echo "BUILD cpp_test"
make -j 4 test || exit -1
export MXNET_ENGINE_INFO=true
for test in tests/cpp/*_test; do
    ./$test || exit -1
done
export MXNET_ENGINE_INFO=false

echo "BUILD python_test"
nosetests --verbose tests/python/unittest || exit -1
nosetests --verbose tests/python/gpu/test_operator_gpu.py || exit -1
nosetests --verbose tests/python/train || exit -1

echo "BUILD python3_test"
nosetests3 --verbose tests/python/unittest || exit -1
nosetests3 --verbose tests/python/gpu/test_operator_gpu.py || exit -1
nosetests3 --verbose tests/python/train || exit -1

echo "BUILD julia_test"
export MXNET_HOME="${PWD}"
/home/ubuntu/julia/bin/julia -e 'try Pkg.clone("MXNet"); catch end; Pkg.checkout("MXNet"); Pkg.build("MXNet"); Pkg.test("MXNet")' || exit -1

echo "BUILD scala_test"
export PATH=$PATH:/opt/apache-maven/bin
make scalapkg || exit -1
make scalatest || exit -1
