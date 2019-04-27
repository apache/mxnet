CC = g++
C = gcc
MKLROOT = /opt/intel/mkl

ifneq ($(USE_INTEL_PATH),)
	MKLROOT = $(USE_INTEL_PATH)/mkl
endif

CFLAGS  = -fpic -O2 -I/opt/intel/mkl/include -c -Wall -Werror  -DMKL_ILP64 -m64  -std=c++11 
LDFLAGS =  -Wl,--start-group -L${MKLROOT}/../compiler/lib/intel64 ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -liomp5 -lpthread -lm -ldl

default: libsparse_matrix.so

libsparse_matrix.so:  sparse_matrix.o
	$(CC) -shared -o libsparse_matrix.so sparse_matrix.o $(LDFLAGS) 

sparse_matrix.o:  sparse_matrix.cc sparse_matrix.h
	$(CC) $(CFLAGS) sparse_matrix.cc 

clean:
	$(RM) libsparse_matrix.so *.o *~
