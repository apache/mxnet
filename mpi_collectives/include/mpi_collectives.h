#ifndef MXNET_MPI_COLLECTIVES_H_
#define MXNET_MPI_COLLECTIVES_H_

#if MXNET_USE_MPI_DIST_KVSTORE

#include <vector>
#include <string>
#include <mxnet/ndarray.h>

namespace mxnet {
namespace kvstore {

int MXMPIGetMpiSize(int *ret);

int MXMPIGetMpiRank(int *ret);

int MXMPIInit();

int MXMPIGetLocalRank(int *ret);

int MXMPIAllReduce(const std::vector<int> &keys,
                   const std::vector<mxnet::NDArray*> &in_values,
                   const std::vector<mxnet::NDArray*> &out_values,
                   int priority);

int MXMPIAllReduceEx(const std::vector<std::string> &keys,
                     const std::vector<mxnet::NDArray*> &in_values,
                     const std::vector<mxnet::NDArray*> &out_values,
                     int priority);

int MXMPIBroadcast(const std::vector<int> &keys,
                   const std::vector<mxnet::NDArray*> &values,
                   int root_rank,
                   int priority);

int MXMPIBroadcastEx(const std::vector<std::string> &keys,
                     const std::vector<mxnet::NDArray*> &values,
                     int root_rank,
                     int priority);

int MXMPIAllGather(const std::vector<int> &keys,
                   const std::vector<mxnet::NDArray*> &values,
                   int priority);

int MXMPIAllGatherEx(const std::vector<std::string> &keys,
                     const std::vector<mxnet::NDArray*> &values,
                     int priority);

}
}
#endif
#endif
