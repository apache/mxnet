#ifndef MXNET_OPERATOR_SPARSE_MATRIX_INL_H_
#define MXNET_OPERATOR_SPARSE_MATRIX_INL_H_


#if (!defined(__INTEL_COMPILER)) & defined(_MSC_VER)
#define SP_INT64 __int64
#define SP_UINT64 unsigned __int64
#else
#define SP_INT64 long long int
#define SP_UINT64 unsigned long long int
#endif


#if defined _WIN32 || defined __CYGWIN__
  #ifdef BUILDING_DLL
    #ifdef __GNUC__
      #define SPM_API_PUBLIC __attribute__ ((dllexport))
    #else
      #define SPM_API_PUBLIC __declspec(dllexport) // Note: actually gcc seems to also supports this syntax.
    #endif
  #else
    #ifdef __GNUC__
      #define SPM_API_PUBLIC __attribute__ ((dllimport))
    #else
      #define SPM_API_PUBLIC __declspec(dllimport) // Note: actually gcc seems to also supports this syntax.
    #endif
  #endif
  #define SPM_API_LOCAL
#else
  #if __GNUC__ >= 4
    #define SPM_API_PUBLIC __attribute__ ((visibility ("default")))
    #define SPM_API_LOCAL  __attribute__ ((visibility ("hidden")))
  #else
    #define SPM_API_PUBLIC
    #define SPM_API_LOCAL
  #endif
#endif



extern "C"
{
	extern SPM_API_PUBLIC bool mkl_DotCsrDnsDns(SP_INT64* rows_start, SP_INT64* col_indx,
		float* values, float* X, float* y, int rows, int cols, int X_columns);

}

#endif //MXNET_OPERATOR_SPARSE_MATRIX_INL_H_