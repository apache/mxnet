#ifndef UTIL_H_
#define UTIL_H_

#if MXNET_USE_MPI_DIST_KVSTORE


#include <stdio.h>

#define DEBUG_ON 0

#if DEBUG_ON
    #define MXMPI_DEBUG(rank, fmt, args...)  \
        do {    \
            printf("rank[%d]:" fmt, rank, ## args); \
        } while(0)
#else
    #define MXMPI_DEBUG(fmt, args...)  do {} while(0)
#endif

/****************************************************
 * The function is used to locate the index of the element
 * in the all equivalent elements.
 *
 * e.g.
 * vec = { 1,6,3,3,1,2,3 }
 *                     ^
 * countNth(vec, 3, 6) = 3
 * vec = { 1,6,3,3,1,2,3 }
 *                 ^
 * countNth(vec, 1, 4) = 2
 ***************************************************/
template <typename T>
size_t countNth(const std::vector<T> &vec,
                const T &key,
                size_t endIdx)
{
  size_t curIdx = 0;
  size_t count = 0;
  for (auto &value: vec) {
    if (curIdx > endIdx) break;
    if (value == key) count++;
    curIdx++;
  }
  return count;
}

#endif
#endif
