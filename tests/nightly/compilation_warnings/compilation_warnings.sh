
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

runme make clean >/dev/null
runme mkdir build
echo "Starting make"
runme time -f "%e" make -j$(nproc) &> build/compile_output.txt
echo "Finished make. Now processing output"
python tests/nightly/compilation_warnings.py build/compile_output.txt
