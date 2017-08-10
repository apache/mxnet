FROM ubuntu
LABEL maintainer "arankhan@amazon.com"

# UPDATE BOX
RUN apt-get update && apt-get -y upgrade

# TOOLCHAIN DEPS
RUN apt-get install -y python python-setuptools python-pip python-dev unzip gfortran
RUN apt-get install -y git nodejs build-essential cmake

# BUILD EMSCRIPTEN
RUN git clone https://github.com/kripken/emscripten.git
RUN git clone https://github.com/kripken/emscripten-fastcomp
RUN cd emscripten-fastcomp && \
git clone https://github.com/kripken/emscripten-fastcomp-clang tools/clang && \
mkdir build && cd build && \
cmake .. -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD="X86;JSBackend" \
-DLLVM_INCLUDE_EXAMPLES=OFF -DLLVM_INCLUDE_TESTS=OFF -DCLANG_INCLUDE_EXAMPLES=OFF \
-DCLANG_INCLUDE_TESTS=OFF && make
