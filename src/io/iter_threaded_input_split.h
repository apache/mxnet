/*!
 *  Copyright (c) 2015 by Contributors
 * \file iter_threaded_input_split.h
 * \brief
 * \author Shuqian Qu
 */
#ifndef MXNET_IO_ITER_THREADED_INPUT_SPLIT_H_
#define MXNET_IO_ITER_THREADED_INPUT_SPLIT_H_
#include <vector>
namespace dmlc {
template <typename DType>
class ThreadedIter;
class InputSplit;
}

namespace mxnet {
namespace io {
/*! \brief common chunk type */
typedef std::vector<char> CommonChunkType;

/*! \brief create input split Threadediter
 *   with ThreadedIter's Next and Recycle iterface, you can consume
 *   InputSplit's data in multi thread
 *
 *   NOTE: different from ThreadedInputSplit, which can only use in single
 *   thread
 *  
 *  \param CommonChunkType common chunk type which contains data buffer
 */
dmlc::ThreadedIter<CommonChunkType>* CreateInputSplitThreadedIter(
    dmlc::InputSplit* base,
    size_t max_capacity = 16, bool ChunkMode = true);

}  // namespace io
}  // namespace mxnet
#endif  // MXNET_IO_ITER_THREADED_INPUT_SPLIT_H_
