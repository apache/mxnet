#ifndef MPI_WRAPPER_H_
#define MPI_WRAPPER_H_

#include <mpi.h>

#include "mxnet/ndarray.h"
#include "mpi_message.pb.h"

template <bool on_gpu, typename DType>
struct MPI_Wrapper
{
  static int Broadcast(mxnet::NDArray *input_array,
                       int root_rank,
                       MPI_Datatype mpi_data_type)
  { return 0; };

  static int AllReduce(mxnet::NDArray *input_array,
                       mxnet::NDArray *output_array,
                       MPI_Datatype mpi_data_type)
  { return 0; };
};

//CPU Implementation
template <typename DType  >
struct MPI_Wrapper <false, DType>
{
  static int Broadcast(mxnet::NDArray *input_array,
                       int root_rank,
                       MPI_Datatype mpi_data_type)
  {
    DType *buf = reinterpret_cast<DType *>(input_array->data().dptr<DType>());
    unsigned int count = input_array->data().Size();
    int ret = MPI_Bcast(buf, count, mpi_data_type, root_rank, MPI_COMM_WORLD);
    return ret;
  }

  static int AllReduce(mxnet::NDArray *input_array,
                       mxnet::NDArray *output_array,
                       MPI_Datatype mpi_data_type)
  {
    DType *send_buf = reinterpret_cast<DType *>(input_array->data().dptr<DType>());
    DType *recv_buf = reinterpret_cast<DType *>(output_array->data().dptr<DType>());
    unsigned int count = input_array->data().Size();
    int ret;
    assert(input_array->data().Size() == output_array->data().Size());

    if (send_buf != recv_buf) {
      ret = MPI_Allreduce((const void *)send_buf, (void *)recv_buf,
                         count, mpi_data_type, MPI_SUM, MPI_COMM_WORLD);
    } else {
      ret = MPI_Allreduce(MPI_IN_PLACE, (void *)recv_buf,
                         count, mpi_data_type, MPI_SUM, MPI_COMM_WORLD);
    }
    return ret;
  }
};

// TODO GPU Implementation
template <typename DType>
struct MPI_Wrapper <true, DType>
{
  static int Broadcast(mxnet::NDArray *input_array,
                       int root_rank,
                       MPI_Datatype mpi_data_type)
  {
    // TODO
    return 0;
  }

  static int AllReduce(mxnet::NDArray *input_array,
                       mxnet::NDArray *output_array,
                       MPI_Datatype mpi_data_type)
  {
    // TODO
    return 0;
  }
};


#endif
