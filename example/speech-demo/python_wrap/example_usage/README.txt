# If not already done, make sure kaldi/src is compiled as shared libraries
cd kaldi/src
./configure --shared
make depend
make

# Copy python_wrap/ to kaldi/src and compile it
cd python_wrap/
make

cd example_usage/
# Add kaldi/src/lib to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=../../lib:$LD_LIBRARY_PATH
python example.py