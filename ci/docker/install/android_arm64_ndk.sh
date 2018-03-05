set -ex
pushd .
export ANDROID_NDK_REVISION=15c
curl -O https://dl.google.com/android/repository/android-ndk-r${ANDROID_NDK_REVISION}-linux-x86_64.zip && \
unzip ./android-ndk-r${ANDROID_NDK_REVISION}-linux-x86_64.zip && \
cd android-ndk-r${ANDROID_NDK_REVISION} && \
./build/tools/make_standalone_toolchain.py \
    --stl=libc++ \
    --arch arm64 \
    --api 21 \
    --install-dir=${CROSS_ROOT} && \

popd