set -ex
# Install clang 3.9 (the same version as in XCode 8.*) and 5.0 (latest major release)
wget -O - http://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - && \
    apt-add-repository "deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial-3.9 main" && \
    apt-add-repository "deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial-5.0 main" && \
    apt-get update && \
    apt-get install -y clang-3.9 clang-5.0 && \
    clang-3.9 --version && \
    clang-5.0 --version