#!/bin/bash
set -e
runme() {
	cmd=$*
	echo "$cmd"
	$cmd
	ret=$?
	if [[ ${ret} != 0 ]]; then
		echo " "
		echo "ERROR: Return value non-zero for: $cmd"
		echo " "
		exit 1
	fi
}

sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get -y install time g++-5
runme make clean >/dev/null
runme mkdir build
echo "Starting make"
cp make/config.mk .
sed -i -e 's/gcc/gcc-5/g' config.mk
sed -i -e 's/g++/g++-5/g' config.mk
runme /usr/bin/time -f "%e" make -j$(nproc) 2>&1 | tee build/compile_output.txt
echo "Finished make. Now processing output"
python tests/nightly/compilation_warnings/process_output.py build/compile_output.txt
