/*!
 *  Copyright (c) 2017 by Contributors
 * \file cast_storage-inl.cuh
 * \brief implementation of cast_storage op on GPU
 */
#ifndef MXNET_OPERATOR_TENSOR_CAST_STORAGE_INL_CUH_
#define MXNET_OPERATOR_TENSOR_CAST_STORAGE_INL_CUH_

#include <mxnet/base.h>
#include <mxnet/operator.h>

#include <cub/cub.cuh>

namespace mxnet {
namespace op {
using mshadow::cuda::kBaseThreadNum;

inline void CastStorageDnsRspImpl(const OpContext& ctx, const gpu& gpu_dev, const TBlob& dns, NDArray* rsp) {
  LOG(FATAL) << "CastStorageDnsRspImpl gpu version is not implemented.";
}

/*!
 * \brief Thread kernel for initializing the indptr in a csr tensor.
 * Parallelized by matrix rows: 1 thread/row
 */
struct FillCsrIndPtrThreadKernel {
  /*!
   * \brief
   * \param tid       global thread id
   * \param indptr    index pointer array of the csr matrix
   * \param dns       dense matrix
   * \param num_rows  number of rows of the dense matrix
   * \param num_cols  number of columns of the dense matrix
   */
  template<typename DType, typename IType>
  __device__ __forceinline__ static void Map(int tid, IType* indptr, const DType* dns,
                                  const int num_rows, const int num_cols) {
    if (tid == 0) {
      indptr[tid] = 0;
    }
    if (tid < num_rows) {
      int nnz = 0;
      const int offset = tid * num_cols;
      for (int j = 0; j < num_cols; ++j) {
        if (dns[offset+j] != 0) {
          nnz++;
        }
      }
      indptr[tid+1] = nnz;
    }
  }
};

/*!
 * \brief Thread kernel for initializing the col_idx and value array of the csr matrix
 * Parallelized by matrix rows: 1 thread/row
 */
struct FillCsrColIdxAndValsThreadKernel {
  /*!
   * \brief
   * \param tid       global thread id
   * \param val       data array of the csr matrix
   * \param col_idx   column index array of the csr matrix
   * \param indptr    index pointer array of the csr matrix
   * \param dns       dense matrix
   * \param num_rows  number of rows of the dense matrix
   * \param num_cols  number of columns of the dense matrix
   */
  template<typename DType, typename IType, typename CType>
  __device__ __forceinline__ static void Map(int tid, DType* val, CType* col_idx,
                                  const IType* indptr, const DType* dns,
                                  const int num_rows, const int num_cols) {
    if (tid < num_rows) {
      const int offset = tid * num_cols;
      int k = indptr[tid];
      for (int j = 0; j < num_cols; ++j) {
        if (dns[offset+j] != 0) {
          val[k] = dns[offset+j];
          col_idx[k] = j;
          ++k;
        }
      }
    }
  }
};

/*!
 * \brief Warp kernel for initializing the indptr in a csr matrix
 * Parallelized by matrix rows: 1 warp/row
 */
struct FillCsrIndPtrWarpKernel {
  template<typename DType, typename IType>
  __device__ __forceinline__ static void Map(int tid, IType* indptr, const DType* dns,
                                  const int num_rows, const int num_cols) {
    typedef cub::WarpReduce<int> WarpReduce;
    const int warps_per_block = kBaseThreadNum / 32;
    __shared__ typename WarpReduce::TempStorage temp_storage[warps_per_block];

    if (tid == 0) {
      indptr[tid] = 0;
    }
    const int warp_id   = tid / 32;          // global warp   id
    const int warp_lane = threadIdx.x / 32;  // local  warp   id within thread block
    const int lane      = tid & (32-1);      // local  thread id within warp
    if (warp_id < num_rows) {
      int lane_nnz = 0;
      const int offset = warp_id * num_cols;
      for (int j = lane; j < num_cols; j+=32) {
        if (dns[offset+j] != 0) {
          lane_nnz++;
        }
      }
      int aggr = WarpReduce(temp_storage[warp_lane]).Sum(lane_nnz);
      if (lane == 0) {
        indptr[warp_id+1] = aggr;
      }
    }
  }
};

/*!
 * \brief Warp kernel for initializing the col_idx and value array of the csr matrix
 * Parallelized by matrix rows: 1 warp/row
 */
struct FillCsrColIdxAndValsWarpKernel {
  template<typename DType, typename IType, typename CType>
  __device__ __forceinline__ static void Map(int tid, DType* val, CType* col_idx,
                                  const IType* indptr, const DType* dns,
                                  const int num_rows, const int num_cols) {
    typedef cub::WarpScan<int> WarpScan;
    const int warps_per_block = kBaseThreadNum / 32;
    __shared__ typename WarpScan::TempStorage temp_storage[warps_per_block];
    __shared__ volatile int warp_nnz[warps_per_block];

    const int warp_id   = tid / 32;          // global warp   id
    const int warp_lane = threadIdx.x / 32;  // local  warp   id within thread block
    const int lane      = tid & (32-1);      // local  thread id within warp
    if (warp_id < num_rows) {
      const int offset = warp_id * num_cols;
      int k = indptr[warp_id];
      int nnz;
      for (int j = lane; j < num_cols+lane; j+=32) {
        nnz = 0;
        if (j < num_cols) {
          if (dns[offset+j] != 0) {
            nnz++;
          }
        }
        if (lane == 31) {
          warp_nnz[warp_lane] = nnz;
        }
        // Compute index each thread has to write to
        WarpScan(temp_storage[warp_lane]).ExclusiveSum(nnz, nnz);
        if (j < num_cols) {
          if (dns[offset+j] != 0) {
            val[k+nnz] = dns[offset+j];
            col_idx[k+nnz] = j;
          }
        }
        if (lane == 31) {
          warp_nnz[warp_lane] += nnz;
        }
        __syncwarp();
        k += warp_nnz[warp_lane];
      }
    }
  }
};

/*!
 * \brief Block kernel for initializing the indptr in a csr tensor.
 * Parallelized by matrix rows: 1 threadBlock/row
 */
struct FillCsrIndPtrBlockKernel {
  template<typename DType, typename IType>
  __device__ __forceinline__ static void Map(int tid, IType* indptr, const DType* dns,
                                  const int num_rows, const int num_cols) {
    typedef cub::BlockReduce<int, kBaseThreadNum> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    if (tid == 0) {
      indptr[tid] = 0;
    }
    if (blockIdx.x < num_rows) {
      int lane_nnz = 0;
      const int offset = blockIdx.x * num_cols;
      for (int j = threadIdx.x; j < num_cols; j+=kBaseThreadNum) {
        if (dns[offset+j] != 0) {
          lane_nnz++;
        }
      }
      int aggr = BlockReduce(temp_storage).Sum(lane_nnz);
      if (threadIdx.x == 0) {
        indptr[blockIdx.x+1] = aggr;
      }
    }
  }
};

/*!
 * \brief Block kernel for initializing the col_idx and value array of the csr matrix
 * Parallelized by matrix rows: 1 threadBlock/row
 */
struct FillCsrColIdxAndValsBlockKernel {
  template<typename DType, typename IType, typename CType>
  __device__ __forceinline__ static void Map(int tid, DType* val, CType* col_idx,
                                  const IType* indptr, const DType* dns,
                                  const int num_rows, const int num_cols) {
    typedef cub::BlockScan<int, kBaseThreadNum> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;
    __shared__ volatile int block_nnz;

    if (blockIdx.x < num_rows) {
      const int offset = blockIdx.x * num_cols;
      int k = indptr[blockIdx.x];
      int nnz;
      for (int j = threadIdx.x; j < num_cols+threadIdx.x; j+=kBaseThreadNum) {
        nnz = 0;
        if (j < num_cols) {
          if (dns[offset+j] != 0) {
            nnz++;
          }
        }
        if (threadIdx.x == kBaseThreadNum-1) {
          block_nnz = nnz;
        }
        // Compute index each thread has to write to
        BlockScan(temp_storage).ExclusiveSum(nnz, nnz);
        if (j < num_cols) {
          if (dns[offset+j] != 0) {
            val[k+nnz] = dns[offset+j];
            col_idx[k+nnz] = j;
          }
        }
        if (threadIdx.x == kBaseThreadNum-1) {
          block_nnz += nnz;
        }
        __syncthreads();
        k += block_nnz;
      }
    }
  }
};

/*!
 * \brief
 * GPU implementation of casting a dense matrix to csr type.
 */
inline void CastStorageDnsCsrImpl(const OpContext& ctx,
                                  const gpu& gpu_dev,
                                  const TBlob& dns,
                                  NDArray* csr) {
  CHECK(csr != nullptr);
  CHECK_EQ(csr->storage_type(), kCSRStorage);
  CHECK_EQ(dns.shape_.ndim(), 2);
  CHECK_EQ(dns.shape_, csr->shape());
  mshadow::Stream<gpu>* s = ctx.get_stream<gpu>();
  MSHADOW_TYPE_SWITCH(dns.type_flag_, DType, {                     // data type
    MSHADOW_IDX_TYPE_SWITCH(csr->aux_type(csr::kIndPtr), IType, {  // indptr type
      MSHADOW_IDX_TYPE_SWITCH(csr->aux_type(csr::kIdx), CType, {   // col_idx type
        const index_t num_rows = dns.shape_[0];
        const index_t num_cols = dns.shape_[1];
        const int threads_per_warp  = 32;
        const int threads_per_block = kBaseThreadNum;
        const int min_num_warps = 512;
        int num_threads;

        csr->CheckAndAllocAuxData(csr::kIndPtr, mshadow::Shape1(num_rows+1));
        IType* indptr = csr->aux_data(csr::kIndPtr).dptr<IType>();
        DType* dns_data = dns.dptr<DType>();

        // Different kernel versions are optimized for different matrix instances
        // (1) 'Thread kernel' (one thread       computing one row)
        // (2) 'Warp kernel'   (one warp         computing one row)
        // (3) 'Block kernel'  (one thread block computing one row)
        const int kernel_version = 0;
        switch (kernel_version) {
          case 1:
            num_threads = num_rows;
            mxnet_op::Kernel<FillCsrIndPtrThreadKernel, gpu>::Launch(s, num_threads,
                indptr, dns_data, num_rows, num_cols);
            break;
          case 2:
            num_threads = num_rows * threads_per_warp;
            mxnet_op::Kernel<FillCsrIndPtrWarpKernel, gpu>::Launch(s, num_threads,
                indptr, dns_data, num_rows, num_cols);
            break;
          case 3:
            num_threads = num_rows * threads_per_block;
            mxnet_op::Kernel<FillCsrIndPtrBlockKernel, gpu>::Launch(s, num_threads,
                indptr, dns_data, num_rows, num_cols);
            break;
          default:
            if (num_cols < threads_per_warp) {
              num_threads = num_rows;
              mxnet_op::Kernel<FillCsrIndPtrThreadKernel, gpu>::Launch(s, num_threads,
                indptr, dns_data, num_rows, num_cols);
            } else if (num_cols < threads_per_block || num_rows > min_num_warps) {
              num_threads = num_rows * threads_per_warp;
              mxnet_op::Kernel<FillCsrIndPtrWarpKernel, gpu>::Launch(s, num_threads,
                indptr, dns_data, num_rows, num_cols);
            } else {
              num_threads = num_rows * threads_per_block;
              mxnet_op::Kernel<FillCsrIndPtrBlockKernel, gpu>::Launch(s, num_threads,
                indptr, dns_data, num_rows, num_cols);
            }
            break;
        }

        // Determine temporary device storage requirements
        void *d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;
        cub::DeviceScan::InclusiveSum(d_temp_storage,
                                      temp_storage_bytes,
                                      indptr,
                                      indptr,
                                      static_cast<int>(num_rows+1),
                                      mshadow::Stream<gpu>::GetStream(s));

        // Allocate temporary storage
        mshadow::Tensor<gpu, 1, char> workspace = ctx.requested[0]
          .get_space_typed<gpu, 1, char>(mshadow::Shape1(temp_storage_bytes), s);
        d_temp_storage = workspace.dptr_;

        // Compute indptr through inclusive prefix sum
        cub::DeviceScan::InclusiveSum(d_temp_storage,
                                      temp_storage_bytes,
                                      indptr,
                                      indptr,
                                      static_cast<int>(num_rows+1),
                                      mshadow::Stream<gpu>::GetStream(s));

        // Receive total number of nnz values from device
        IType nnz = 0;
        CUDA_CALL(cudaMemcpy(&nnz, &(indptr[num_rows]), sizeof(IType), cudaMemcpyDeviceToHost));

        // Allocate column index array and data array of the csr matrix
        csr->CheckAndAllocAuxData(csr::kIdx, mshadow::Shape1(static_cast<index_t>(nnz)));
        csr->CheckAndAllocData(mshadow::Shape1(static_cast<index_t>(nnz)));

        // Compute and fill column index array and data array of the csr matrix
        switch (kernel_version) {
          case 1:
            num_threads = num_rows;
            mxnet_op::Kernel<FillCsrColIdxAndValsThreadKernel, gpu>::Launch(s, num_threads,
                csr->data().dptr<DType>(), csr->aux_data(csr::kIdx).dptr<CType>(),
                indptr, dns_data, num_rows, num_cols);
            break;
          case 2:
            num_threads = num_rows * threads_per_warp;
            mxnet_op::Kernel<FillCsrColIdxAndValsWarpKernel, gpu>::Launch(s, num_threads,
                csr->data().dptr<DType>(), csr->aux_data(csr::kIdx).dptr<CType>(),
                indptr, dns_data, num_rows, num_cols);
            break;
          case 3:
            num_threads = num_rows * threads_per_block;
            mxnet_op::Kernel<FillCsrColIdxAndValsBlockKernel, gpu>::Launch(s, num_threads,
                csr->data().dptr<DType>(), csr->aux_data(csr::kIdx).dptr<CType>(),
                indptr, dns_data, num_rows, num_cols);
            break;
          default:
            if (num_cols < threads_per_warp) {
              num_threads = num_rows;
              mxnet_op::Kernel<FillCsrColIdxAndValsThreadKernel, gpu>::Launch(s, num_threads,
                csr->data().dptr<DType>(), csr->aux_data(csr::kIdx).dptr<CType>(),
                indptr, dns_data, num_rows, num_cols);
            } else if (num_cols < threads_per_block || num_rows > min_num_warps) {
              num_threads = num_rows * threads_per_warp;
              mxnet_op::Kernel<FillCsrColIdxAndValsWarpKernel, gpu>::Launch(s, num_threads,
                csr->data().dptr<DType>(), csr->aux_data(csr::kIdx).dptr<CType>(),
                indptr, dns_data, num_rows, num_cols);
            } else {
              num_threads = num_rows * threads_per_block;
              mxnet_op::Kernel<FillCsrColIdxAndValsBlockKernel, gpu>::Launch(s, num_threads,
                csr->data().dptr<DType>(), csr->aux_data(csr::kIdx).dptr<CType>(),
                indptr, dns_data, num_rows, num_cols);
            }
            break;
        }
      });
    });
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_TENSOR_CAST_STORAGE_INL_CUH_
