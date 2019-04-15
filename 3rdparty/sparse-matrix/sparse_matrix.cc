#include <iostream>
#include <string>
#include <fstream>
#include <mkl_spblas.h>
#include "sparse_matrix.h"



bool mkl_DotCsrDnsDns(SP_INT64* rows_start, SP_INT64* col_indx,
	float* values, float* X, float* y,
	int rows, int cols, int X_columns)
{

	sparse_index_base_t indexing = SPARSE_INDEX_BASE_ZERO;
	sparse_status_t status;
	sparse_matrix_t A = NULL;
	sparse_layout_t layout = SPARSE_LAYOUT_ROW_MAJOR;
	float one, zero;
	one = (float)1.0;
	zero = (float)0.0;

	MKL_INT* rows_end = rows_start + 1;
	status = mkl_sparse_s_create_csr(&A, indexing, rows, cols, rows_start, rows_end, col_indx, values);

  if (status != SPARSE_STATUS_SUCCESS)
  {
    std::cout << "mkl_sparse_s_create_csr status :" << status << std::endl;
    return false;
  }
	sparse_operation_t operation = SPARSE_OPERATION_NON_TRANSPOSE;
	struct matrix_descr descrA;
	descrA.type = SPARSE_MATRIX_TYPE_GENERAL;

	status = mkl_sparse_s_mm(operation, one, A, descrA, layout, X, X_columns, X_columns, zero, y, X_columns);
  if (status != SPARSE_STATUS_SUCCESS)
  {
    std::cout << "mkl_sparse_s_create_csr status :" << status << std::endl;
    return false;
  }
	
	mkl_sparse_destroy(A);
	
	return true;

}
