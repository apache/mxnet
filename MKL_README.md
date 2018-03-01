# Full MKL Installation

## Build/Install MXNet with a full MKL installation:
Installing and enabling the full MKL installation enables MKL support for all operators under the linalg namespace.

  1. Download and install the latest full MKL version following instructions on the [intel website.](https://software.intel.com/en-us/articles/intel-mkl-111-install-guide)

  2. Set USE_BLAS=mkl in make/config.mk

        1.1 Set ADD_LDFLAGS=-L<path/to/mkl/lib/folder> (ex. ADD_LDFLAGS=-L/opt/intel/compilers_and_libraries_2018.0.128/linux/mkl/lib)

        1.1 Set ADD_CFLAGS=-I<path/to/mkl/include/folder> (ex. ADD_CFLAGS=-L/opt/intel/compilers_and_libraries_2018.0.128/linux/mkl/include)

  3. Run 'make -j ${nproc}'

  4. Navigate into the python directory

  5. Run 'sudo python setup.py install'

