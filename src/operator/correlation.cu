/*!
 * Copyright [2016] <Contributors>
 * \file Correation.cu
 * \brief  Correlation operator
 * \author Xu Dong
*/
#include "./correlation-inl.h"
#include <mshadow/tensor.h>
#include <mshadow/cuda/reduce.cuh>
#include <algorithm>
#include <vector>

#define ROUND_OFF 50000
#define WARPS_PER_BLOCK 1
#define THREADS_PER_WARP 32
#define CORRELATION_CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)
#define CUDA_KERNEL_LOOP(i, n) \
for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n); \
      i += blockDim.x * gridDim.x)
namespace mshadow {
namespace cuda {
// == Correlation Kernel
template <typename Dtype>
__global__ void CorrelateData(const int nthreads, int num, int topwidth,
  int topheight, int topchannels, int topcount,
  int max_displacement, int neighborhood_grid_radius,
  int neighborhood_grid_width, int kernel_radius, int kernel_size, int stride1, int stride2,
  int bottomwidth, int bottomheight, int bottomchannels,
  const Dtype *bottom0, const Dtype *bottom1, Dtype *top) {
  extern __shared__ char patch_data_char[];
  Dtype *patch_data = reinterpret_cast<Dtype *>(patch_data_char);
  //  First (upper left) position of kernel upper-left corner
  //  in current center position of neighborhood in image 1
  int x1 = blockIdx.x * stride1 + max_displacement;
  int y1 = blockIdx.y * stride1 + max_displacement;
  int item = blockIdx.z;
  int ch_off = threadIdx.x;
  //  Load 3D patch into shared shared memory
  for (int j = 0; j < kernel_size; j++) {  //  HEIGHT
    for (int i = 0; i < kernel_size; i++) {  //  WIDTH
      int ji_off = ((j * kernel_size) + i) * bottomchannels;
      for (int ch = ch_off; ch < bottomchannels; ch += (THREADS_PER_WARP * WARPS_PER_BLOCK))  {
          //  CHANNELS
          int idx1 = ((item * bottomheight + y1+j) * bottomwidth + x1+i) * bottomchannels + ch;
          int idxPatchData = ji_off + ch;
          patch_data[idxPatchData] = bottom0[idx1];
      }
    }
  }
  __syncthreads();
  __shared__ Dtype sum[THREADS_PER_WARP * WARPS_PER_BLOCK];
  //  Compute correlation
  for (int top_channel = 0; top_channel < topchannels; top_channel++) {
    sum[ch_off] = 0;
    int s2o = (top_channel % neighborhood_grid_width - neighborhood_grid_radius) * stride2;
    int s2p = (top_channel / neighborhood_grid_width - neighborhood_grid_radius) * stride2;
    for (int j = 0; j < kernel_size; j++) {  //  HEIGHT
      for (int i = 0; i < kernel_size; i++) {  //  WIDTH
        int ji_off = ((j * kernel_size) + i) * bottomchannels;
        for (int ch = ch_off; ch < bottomchannels; ch += (THREADS_PER_WARP * WARPS_PER_BLOCK)) {
          //  CHANNELS
          int x2 = x1 + s2o;
          int y2 = y1 + s2p;
          int idxPatchData = ji_off + ch;
          int idx2 = ((item * bottomheight + y2 + j) * bottomwidth + x2 + i) * bottomchannels + ch;
          sum[ch_off] += patch_data[idxPatchData] * bottom1[idx2];
        }
      }
    }
    __syncthreads();
    if (ch_off == 0) {
        Dtype total_sum = 0;
        for (int idx = 0; idx < THREADS_PER_WARP * WARPS_PER_BLOCK; idx++) {
            total_sum += sum[idx];
        }
        const int sumelems = kernel_size * kernel_size * bottomchannels;
        const int index = ((top_channel * topheight + blockIdx.y) * topwidth) + blockIdx.x;
        top[index + item*topcount] = total_sum / static_cast<float>(sumelems);
    }  //  Aggregate result of  different threads
  }
}
//  == Correlation Backward Pass Kernel (For data1)
template <typename Dtype>
__global__ void CorrelateDataBackward0(const int nthreads, int num, int item,
  int topwidth, int topheight, int topchannels,
  int max_displacement, int neighborhood_grid_radius,
  int neighborhood_grid_width, int kernel_radius, int stride1, int stride2,
  int bottomwidth, int bottomheight, int pbottomwidth, int pbottomheight,
  int bottomchannels, int bottomcount, int pad_size,
  Dtype *bottom0diff, const Dtype *bottom1, const Dtype *topdiff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int n = index % bottomchannels;  //  channels
    int l = (index / bottomchannels) % bottomwidth + pad_size;  //  w-pos
    int m = (index / bottomchannels / bottomwidth) % bottomheight + pad_size;  //  h-pos
    //  Get X,Y ranges and clamp
    //  round_off is a trick to enable integer division with ceil, even for negative numbers
    //  We use a large offset, for the inner part not to become negative.
    const int round_off = ROUND_OFF;
    const int round_off_s1 = stride1 * round_off;
    //  We add round_off before_s1 the int division and subtract round_off after it,
    //  to ensure the formula matches ceil behavior:
    int xmin = (l - 2*kernel_radius - max_displacement + round_off_s1 - 1)\
     / stride1 + 1 - round_off;  //  ceil (l - 2*kernel_radius - max_displacement) / stride1
    int ymin = (m - 2*kernel_radius - max_displacement + round_off_s1 - 1)\
     / stride1 + 1 - round_off;  //  ceil (l - 2*kernel_radius - max_displacement) / stride1
    //  Same here:
    int xmax = (l - max_displacement + round_off_s1) / stride1 - round_off;
    //  floor (l - max_displacement) / stride1
    int ymax = (m - max_displacement + round_off_s1) / stride1 - round_off;
    //  floor (m - max_displacement) / stride1
    Dtype sum = 0;
    if (xmax >= 0 && ymax >= 0 && (xmin <= topwidth-1) && (ymin <= topheight-1)) {
        xmin = max(0, xmin);
        xmax = min(topwidth-1, xmax);
        ymin = max(0, ymin);
        ymax = min(topheight-1, ymax);
        for (int p = -neighborhood_grid_radius; p <= neighborhood_grid_radius; p++) {
          for (int o = -neighborhood_grid_radius; o <= neighborhood_grid_radius; o++) {
            //  Get bottom1 data:
            int s2o = stride2 * o;
            int s2p = stride2 * p;
            int idxbot1 = ((item * pbottomheight + (m + s2p)) * pbottomwidth + (l + s2o))\
             * bottomchannels + n;
            Dtype bot1tmp = bottom1[idxbot1];  // bottom1[l+s2o,m+s2p,n]
            //  Index offset for topdiff in following loops:
            int op = (p+neighborhood_grid_radius) * neighborhood_grid_width\
             + (o + neighborhood_grid_radius);  //  index [o,p]
            int idxopoffset = (item * topchannels + op);
            for (int y = ymin; y <= ymax; y++) {
              for (int x = xmin; x <= xmax; x++) {
                int idxtopdiff = (idxopoffset * topheight + y) * topwidth + x;  //  topdiff[x,y,o,p]
                sum += topdiff[idxtopdiff] * bot1tmp;
              }
            }
          }
        }
    }
    const int sumelems = (kernel_radius * 2 + 1) * (kernel_radius * 2+1) * bottomchannels;
    const int bot0index = ((n * bottomheight) + (m-pad_size)) * bottomwidth + (l-pad_size);
    bottom0diff[bot0index + item * bottomcount] = sum / static_cast<float>(sumelems);
  }
}
// == Correlation Backward Pass Kernel (For Blob 1)
template <typename Dtype>
__global__ void CorrelateDataBackward1(const int nthreads,
  int num, int item, int topwidth, int topheight, int topchannels,
  int max_displacement, int neighborhood_grid_radius,
  int neighborhood_grid_width, int kernel_radius, int stride1, int stride2,
  int bottomwidth, int bottomheight, int pbottomwidth, int pbottomheight,
  int bottomchannels, int bottomcount, int pad_size,
  const Dtype *bottom0, Dtype *bottom1diff, const Dtype *topdiff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    //  int l = index % bottomwidth + pad_size; //w-pos
    //  int m = (index / bottomwidth) % bottomheight + pad_size; //  h-pos
    //  int n = (index / bottomwidth / bottomheight) % bottomchannels; //  channels
    int n = index % bottomchannels;  //  channels
    int l = (index / bottomchannels) % bottomwidth + pad_size;  //  w-pos
    int m = (index / bottomchannels / bottomwidth) % bottomheight + pad_size;  //  h-pos
    //  round_off is a trick to enable integer division with ceil, even for negative numbers
    //  We use a large offset, for the inner part not to become negative.
    const int round_off = ROUND_OFF;
    const int round_off_s1 = stride1 * round_off;
    Dtype sum = 0;
    for (int p = -neighborhood_grid_radius; p <= neighborhood_grid_radius; p++) {
      for (int o = -neighborhood_grid_radius; o <= neighborhood_grid_radius; o++) {
        int s2o = stride2 * o;
        int s2p = stride2 * p;
        //  Get X,Y ranges and clamp
        //  We add round_off before_s1 the int division and subtract round_off after it,
        //  to ensure the formula matches ceil behavior:
        int xmin = (l - 2*kernel_radius - max_displacement - s2o + round_off_s1 - 1)\
         / stride1 + 1 - round_off;
         // ceil (l - 2*kernel_radius - max_displacement - s2o) / stride1
        int ymin = (m - 2*kernel_radius - max_displacement - s2p + round_off_s1 - 1)\
         / stride1 + 1 - round_off;
        // ceil (l - 2*kernel_radius - max_displacement - s2o) / stride1
        //  Same here:
        int xmax = (l - max_displacement - s2o + round_off_s1) / stride1 - round_off;
        //  floor (l - max_displacement - s2o) / stride1
        int ymax = (m - max_displacement - s2p + round_off_s1) / stride1 - round_off;
        //  floor (m - max_displacement - s2p) / stride1
        if (xmax >= 0 && ymax >= 0 && (xmin <= topwidth - 1) && (ymin <= topheight - 1)) {
            xmin = max(0, xmin);
            xmax = min(topwidth-1, xmax);
            ymin = max(0, ymin);
            ymax = min(topheight-1, ymax);
            //  Get bottom0 data:
            int idxbot0 = ((item * pbottomheight + (m - s2p)) \
            * pbottomwidth + (l - s2o)) * bottomchannels + n;
            Dtype bot0tmp = bottom0[idxbot0];  //  bottom1[l+s2o,m+s2p,n]
            //  Index offset for topdiff in following loops:
            int op = (p+neighborhood_grid_radius) * \
            neighborhood_grid_width + (o+neighborhood_grid_radius);  //  index [o,p]
            int idxOpOffset = (item * topchannels + op);
            for (int y = ymin; y <= ymax; y++) {
              for (int x = xmin; x <= xmax; x++) {
                int idxtopdiff = (idxOpOffset * topheight + y)\
                 * topwidth + x;  //  topdiff[x,y,o,p]
                sum += topdiff[idxtopdiff] * bot0tmp;
              }
            }
        }
      }
    }
    const int sumelems = (kernel_radius*2+1)*(kernel_radius*2+1)*bottomchannels;
    const int bot1index = ((n * bottomheight) + (m - pad_size)) * bottomwidth + (l - pad_size);
    bottom1diff[bot1index + item * bottomcount] = sum / static_cast<float>(sumelems);
  }
}
// == Correlation Kernel Subtraction
template <typename Dtype>
__global__ void CorrelateDataSubtract(const int nthreads, int num, int item,
  int topwidth, int topheight, int topchannels, int topcount,
  int max_displacement, int neighborhood_grid_radius,
  int neighborhood_grid_width, int kernel_radius, int stride1, int stride2,
  int bottomwidth, int bottomheight, int bottomchannels,
  const Dtype *bottom0, const Dtype *bottom1, Dtype *top) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int x = index % topwidth;  //  w-pos
    int y = (index / topwidth) % topheight;  //  h-pos
    int c = (index / topwidth / topheight) % topchannels;  //  channels
    //  Offset of patch in image 2
    int s2o = (c % neighborhood_grid_width - neighborhood_grid_radius) * stride2;
    int s2p = (c / neighborhood_grid_width - neighborhood_grid_radius) * stride2;
    //  First (upper left) position of kernel center in current neighborhood in image 1
    int x1 = x*stride1 + kernel_radius + max_displacement;
    int y1 = y*stride1 + kernel_radius + max_displacement;
    //  Iterate through 3D patch
    Dtype sum = 0;
    for (int j = -kernel_radius; j <= kernel_radius; j++) {  //  HEIGHT
      for (int i = -kernel_radius; i <= kernel_radius; i++) {  //  WIDTH
        for (int l = 0; l < bottomchannels; l++) {  //  CHANNELS
          //  Calculate position in image 2
          int x2 = x1 + s2o;
          int y2 = y1 + s2p;
          //  Indices in bottom data: (CH=l,W=x2,H=y2,N)
          int idx1 = ((item * bottomheight + y1 + j) * bottomwidth + x1 + i) \
          * bottomchannels + l;
          int idx2 = ((item * bottomheight + y2 + j) * bottomwidth + x2 + i) \
          * bottomchannels + l;
          //  Do the correlation:
          sum += fabsf(bottom0[idx1] - bottom1[idx2]);
        }
      }
    }
    const int sumelems = (kernel_radius * 2 + 1) * (kernel_radius * 2 + 1) * bottomchannels;
    top[index + item * topcount] = sum / static_cast<float>(sumelems);
  }
}
//  == Correlation Backward Pass Kernel (For Blob 0)
template <typename Dtype>
__global__ void CorrelateDataBackward0Subtract(const int nthreads, int num,
  int item, int topwidth, int topheight, int topchannels,
  int max_displacement, int neighborhood_grid_radius,
  int neighborhood_grid_width, int kernel_radius,
  int stride1, int stride2, int bottomwidth, int bottomheight,
  int pbottomwidth, int pbottomheight,
  int bottomchannels, int bottomcount, int pad_size,
  Dtype *bottom0diff, const Dtype *bottom0, const Dtype *bottom1, const Dtype *topdiff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int n = index % bottomchannels;  //  channels
    int l = (index / bottomchannels) % bottomwidth + pad_size;  //  w-pos
    int m = (index / bottomchannels / bottomwidth) % bottomheight + pad_size;  //  h-pos
    //  Get X,Y ranges and clamp
    //  round_off is a trick to enable integer division with ceil, even for negative numbers
    //  We use a large offset, for the inner part not to become negative.
    const int round_off = ROUND_OFF;
    const int round_off_s1 = stride1 * round_off;
    int idxbot0 = ((item * pbottomheight + m) * pbottomwidth + l)\
             * bottomchannels + n;
    //  We add round_off before_s1 the int division and subtract round_off after it,
    //  to ensure the formula matches ceil behavior:
    int xmin = (l - 2*kernel_radius - max_displacement + round_off_s1 - 1)\
     / stride1 + 1 - round_off;  //  ceil (l - 2*kernel_radius - max_displacement) / stride1
    int ymin = (m - 2*kernel_radius - max_displacement + round_off_s1 - 1)\
     / stride1 + 1 - round_off;  //  ceil (l - 2*kernel_radius - max_displacement) / stride1
    //  Same here:
    int xmax = (l - max_displacement + round_off_s1) / stride1 - round_off;
    //  floor (l - max_displacement) / stride1
    int ymax = (m - max_displacement + round_off_s1) / stride1 - round_off;
    //  floor (m - max_displacement) / stride1
    Dtype sum = 0;
    if (xmax >= 0 && ymax >= 0 && (xmin <= topwidth-1) && (ymin <= topheight-1)) {
        xmin = max(0, xmin);
        xmax = min(topwidth-1, xmax);
        ymin = max(0, ymin);
        ymax = min(topheight-1, ymax);
        for (int p = -neighborhood_grid_radius; p <= neighborhood_grid_radius; p++) {
          for (int o = -neighborhood_grid_radius; o <= neighborhood_grid_radius; o++) {
            //  Get bottom1 data:
            int s2o = stride2 * o;
            int s2p = stride2 * p;
            int idxbot1 = ((item * pbottomheight + (m+s2p)) * pbottomwidth\
             + (l+s2o)) * bottomchannels + n;
            Dtype bot0tmp = bottom0[idxbot0];
            Dtype bot1tmp = bottom1[idxbot1];
            Dtype sign = (bot0tmp >= bot1tmp) ? Dtype(1.0) : Dtype(-1.0);
            //  Index offset for topdiff in following loops:
            int op = (p+neighborhood_grid_radius) * neighborhood_grid_width\
             + (o + neighborhood_grid_radius);  //  index [o,p]
            int idxopoffset = (item * topchannels + op);
            for (int y = ymin; y <= ymax; y++) {
              for (int x = xmin; x <= xmax; x++) {
                int idxtopdiff = (idxopoffset * topheight + y) * topwidth + x;  //  topdiff[x,y,o,p]
                sum += topdiff[idxtopdiff] * sign;
              }
            }
          }
        }
    }
    const int sumelems = (kernel_radius * 2 + 1) * (kernel_radius * 2+1) * bottomchannels;
    const int bot0index = ((n * bottomheight) + (m-pad_size)) * bottomwidth + (l-pad_size);
    bottom0diff[bot0index + item * bottomcount] = sum / static_cast<float>(sumelems);
  }
}
//  == Correlation Backward Pass Kernel (For Blob 1)
template <typename Dtype>
__global__ void CorrelateDataBackward1Subtract(const int nthreads, int num,
  int item, int topwidth, int topheight, int topchannels,
  int max_displacement, int neighborhood_grid_radius,
  int neighborhood_grid_width, int kernel_radius,
  int stride1, int stride2, int bottomwidth, int bottomheight,
  int pbottomwidth, int pbottomheight, int bottomchannels,
  int bottomcount, int pad_size, const Dtype *bottom0,
  const Dtype *bottom1, Dtype *bottom1diff, const Dtype *topdiff) {
    CUDA_KERNEL_LOOP(index, nthreads) {
    //  int l = index % bottomwidth + pad_size; //w-pos
    //  int m = (index / bottomwidth) % bottomheight + pad_size; //  h-pos
    //  int n = (index / bottomwidth / bottomheight) % bottomchannels; //  channels
    int n = index % bottomchannels;  //  channels
    int l = (index / bottomchannels) % bottomwidth + pad_size;  //  w-pos
    int m = (index / bottomchannels / bottomwidth) % bottomheight + pad_size;  //  h-pos
    //  round_off is a trick to enable integer division with ceil, even for negative numbers
    //  We use a large offset, for the inner part not to become negative.
    const int round_off = ROUND_OFF;
    const int round_off_s1 = stride1 * round_off;
    Dtype sum = 0;
    int idxbot1 = ((item * pbottomheight + m) * pbottomwidth + l)\
             * bottomchannels + n;
    for (int p = -neighborhood_grid_radius; p <= neighborhood_grid_radius; p++) {
      for (int o = -neighborhood_grid_radius; o <= neighborhood_grid_radius; o++) {
        int s2o = stride2 * o;
        int s2p = stride2 * p;
        //  Get X,Y ranges and clamp
        //  We add round_off before_s1 the int division and subtract round_off after it,
        //  to ensure the formula matches ceil behavior:
        int xmin = (l - 2*kernel_radius - max_displacement - s2o + round_off_s1 - 1)\
         / stride1 + 1 - round_off;
         // ceil (l - 2*kernel_radius - max_displacement - s2o) / stride1
        int ymin = (m - 2*kernel_radius - max_displacement - s2p + round_off_s1 - 1)\
         / stride1 + 1 - round_off;
        // ceil (l - 2*kernel_radius - max_displacement - s2o) / stride1
        //  Same here:
        int xmax = (l - max_displacement - s2o + round_off_s1) / stride1 - round_off;
        //  floor (l - max_displacement - s2o) / stride1
        int ymax = (m - max_displacement - s2p + round_off_s1) / stride1 - round_off;
        //  floor (m - max_displacement - s2p) / stride1
        if (xmax >= 0 && ymax >= 0 && (xmin <= topwidth - 1) && (ymin <= topheight - 1)) {
            xmin = max(0, xmin);
            xmax = min(topwidth-1, xmax);
            ymin = max(0, ymin);
            ymax = min(topheight-1, ymax);
            //  Get bottom0 data:
            int idxbot0 = ((item * pbottomheight + (m - s2p)) * pbottomwidth + (l - s2o))\
             * bottomchannels + n;
            //  bottom0[l+s2o,m+s2p,n]
            Dtype bot0tmp = bottom0[idxbot0];
            Dtype bot1tmp = bottom1[idxbot1];
            Dtype sign = (bot0tmp >= bot1tmp) ? Dtype(-1.0) : Dtype(1.0);
            //  Index offset for topdiff in following loops:
            int op = (p+neighborhood_grid_radius) * \
            neighborhood_grid_width + (o+neighborhood_grid_radius);  //  index [o,p]
            int idxOpOffset = (item * topchannels + op);
            for (int y = ymin; y <= ymax; y++) {
              for (int x = xmin; x <= xmax; x++) {
                int idxtopdiff = (idxOpOffset * topheight + y)\
                 * topwidth + x;  //  topdiff[x,y,o,p]
                sum += topdiff[idxtopdiff] * sign;
              }
            }
        }
      }
    }
    const int sumelems = (kernel_radius*2+1)*(kernel_radius*2+1)*bottomchannels;
    const int bot1index = ((n * bottomheight) + (m - pad_size)) * bottomwidth + (l - pad_size);
    bottom1diff[bot1index + item * bottomcount] = sum / static_cast<float>(sumelems);
  }
}
//  == Forward
//  == Dimension rearrangement Kernel
template <typename Dtype>
__global__ void blob_rearrange_kernel2(const Dtype* in, Dtype* out, int num,
int channels, int width, int height, int widthheight, int padding, int pwidthheight) {
    //  change shape from [batchsize,channel,y,x] to [batchsize,y,x,channel]
    int xy = blockIdx.x * blockDim.x + threadIdx.x;
    if (xy >= widthheight )
        return;
    int ch = blockIdx.y;
    int n  = blockIdx.z;
    Dtype value = in[(n * channels + ch) * widthheight + xy];
    __syncthreads();
    int xpad  = (xy % width + padding);
    int ypad  = (xy / width + padding);
    int xypad = ypad * (width + 2 * padding) + xpad;
    out[(n * pwidthheight + xypad) * channels + ch] = value;
}
template <typename Dtype>
void Forward_gpu(
      const Tensor<gpu, 4, Dtype> &out,
      const Tensor<gpu, 4, Dtype> &data1,
      const Tensor<gpu, 4, Dtype> &data2,
      const Tensor<gpu, 4, Dtype> &tmp1,
      const Tensor<gpu, 4, Dtype> &tmp2,
      int top_channels_, int top_height_, int top_width_, int pad_size_,
      bool is_multiply, int max_displacement_, int kernel_size_,
      int neighborhood_grid_radius_, int neighborhood_grid_width_,
      int  kernel_radius_, int stride1_, int stride2_, cudaStream_t stream,
      cudaStream_t stream_tmp1, cudaStream_t stream_tmp2) {
    const Dtype *bottom_data1 = data1.dptr_;
    const Dtype *bottom_data2 = data2.dptr_;
    Dtype *rbot1 = tmp1.dptr_;
    Dtype *rbot2 = tmp2.dptr_;
    Dtype *top = out.dptr_;
    const int bnum = data1.size(0);
    const int bchannels = data1.size(1);
    const int bheight = data1.size(2);
    const int bwidth = data1.size(3);
    const int bwidthheight = bwidth * bheight;
    const int topcount = top_width_ * top_height_ * top_channels_;
    dim3 threadsPerBlock(THREADS_PER_WARP * WARPS_PER_BLOCK);
    int threads_per_block = 16;
    dim3 totalBlocksRearr((bwidthheight - 1) / threads_per_block + 1, bchannels, bnum);
    const int pwidthheight = (bwidth + 2 * pad_size_) * (bheight + 2 * pad_size_);
    blob_rearrange_kernel2<Dtype><<<totalBlocksRearr, threads_per_block, 0, stream_tmp1>>>
    (bottom_data1, rbot1, bnum, bchannels, bwidth, bheight, bwidthheight, pad_size_, pwidthheight);
    blob_rearrange_kernel2<Dtype><<<totalBlocksRearr, threads_per_block, 0, stream_tmp2>>>
    (bottom_data2, rbot2, bnum, bchannels, bwidth, bheight, bwidthheight, pad_size_, pwidthheight);
    const int num = bnum;
    const int channels = bchannels;
    const int height = bheight + 2 * pad_size_;
    const int width = bwidth + 2 * pad_size_;
    const int shared_memory_per_block = (kernel_size_ * kernel_size_) * bchannels;
    if (is_multiply == true) {
        //  CorrelationLayer
        int topThreadCount = topcount;
        dim3 totalBlocksCorr(top_width_, top_height_, num);
        CorrelateData<Dtype><<<totalBlocksCorr, threadsPerBlock,
        shared_memory_per_block * sizeof(Dtype), stream>>>(
            topThreadCount,
            num, top_width_, top_height_, top_channels_, topcount,
            max_displacement_, neighborhood_grid_radius_,
            neighborhood_grid_width_, kernel_radius_, kernel_size_,
            stride1_, stride2_,
            width, height, channels,
            rbot1, rbot2, top);
        CORRELATION_CUDA_CHECK(cudaPeekAtLastError());
    } else {
        //  CorrelationLayer
        for (int n = 0; n < num; n++) {
            int topThreadCount = topcount;
            const int gridSize = (topThreadCount + kMaxThreadsPerBlock - 1)\
             / kMaxThreadsPerBlock;
            CorrelateDataSubtract<Dtype><<<gridSize, kMaxThreadsPerBlock, 0, stream>>>(
                topThreadCount,
                num, n, top_width_, top_height_, top_channels_, topcount,
                max_displacement_, neighborhood_grid_radius_,
                neighborhood_grid_width_, kernel_radius_,
                stride1_, stride2_, width, height, channels, rbot1, rbot2, top);
         CORRELATION_CUDA_CHECK(cudaPeekAtLastError());
        }
    }
}
template <typename Dtype>
void Backward_gpu(
       const Tensor<gpu, 4, Dtype> &out_grad,
      const Tensor<gpu, 4, Dtype> &in_grad1,
      const Tensor<gpu, 4, Dtype> &in_grad2,
      const Tensor<gpu, 4, Dtype> &tmp1,
      const Tensor<gpu, 4, Dtype> &tmp2,
      int top_channels_, int top_height_,
      int top_width_, int pad_size_, bool is_multiply,
      int max_displacement_, int kernel_size_,
      int neighborhood_grid_radius_, int neighborhood_grid_width_,
      int  kernel_radius_, int stride1_, int stride2_,
      cudaStream_t stream0, cudaStream_t stream1,
      int num, int channels, int height, int width) {
    //  Get top diff, compute bottom diff
    const Dtype* top_diff = out_grad.dptr_;
    Dtype* bottom0_diff = in_grad1.dptr_;
    Dtype* bottom1_diff = in_grad2.dptr_;
    const Dtype* rbot1 = tmp1.dptr_;
    const Dtype* rbot2 = tmp2.dptr_;
    const int paddedheight = height + 2 * pad_size_;
    const int paddedwidth = width + 2 * pad_size_;
    const int bottomcount = channels * height * width;
    int botThreadCount = bottomcount;
    const int gridSize = (botThreadCount + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
    //  CorrelationLayerBackward
    if (is_multiply == true) {
        //  == Run kernel Backward 0
        dim3 totalBlocksBackward0(width, height, channels * num);  //  First dim is fastest
        const int buffer_size_backw0 = \
        (static_cast<int>(ceil(static_cast<float>(2 * kernel_radius_)\
         / static_cast<float>(stride1_))) + 1) * top_channels_;
        //  == Run kernel Backward 0
        for (int n = 0; n < num; n++) {
        CorrelateDataBackward0<Dtype><<<gridSize, kMaxThreadsPerBlock, 0, stream0>>>(
            botThreadCount,
            num, n, top_width_, top_height_, top_channels_,
            max_displacement_, neighborhood_grid_radius_, neighborhood_grid_width_, kernel_radius_,
            stride1_, stride2_,
            width, height, paddedwidth, paddedheight, channels, bottomcount, pad_size_,
            bottom0_diff, rbot2, top_diff);
        CORRELATION_CUDA_CHECK(cudaPeekAtLastError());
        }
        //  == Run kernel Backward 1
        for (int n = 0; n < num; n++) {
        CorrelateDataBackward1<Dtype><<<gridSize, kMaxThreadsPerBlock, 0, stream1>>>(
            botThreadCount,
            num, n, top_width_, top_height_, top_channels_,
            max_displacement_, neighborhood_grid_radius_, neighborhood_grid_width_, kernel_radius_,
            stride1_, stride2_,
            width, height, paddedwidth, paddedheight, channels, bottomcount, pad_size_,
            rbot1, bottom1_diff, top_diff);
       CORRELATION_CUDA_CHECK(cudaPeekAtLastError());
        }
    } else  {
        for (int n = 0; n < num; n++) {
        //  Bottom0:
        CorrelateDataBackward0Subtract<Dtype><<<gridSize, kMaxThreadsPerBlock, 0, stream0>>>(
            botThreadCount,
            num, n, top_width_, top_height_, top_channels_,
            max_displacement_, neighborhood_grid_radius_, neighborhood_grid_width_, kernel_radius_,
            stride1_, stride2_,
            width, height, paddedwidth, paddedheight, channels, bottomcount, pad_size_,
            bottom0_diff, rbot1, rbot2, top_diff);
        CORRELATION_CUDA_CHECK(cudaPeekAtLastError());
        }
        for (int n = 0; n < num; n++) {
        //  Bottom1:
        CorrelateDataBackward1Subtract<Dtype><<<gridSize, kMaxThreadsPerBlock, 0, stream1>>>(
            botThreadCount,
            num, n, top_width_, top_height_, top_channels_,
            max_displacement_, neighborhood_grid_radius_, neighborhood_grid_width_, kernel_radius_,
            stride1_, stride2_,
            width, height, paddedwidth, paddedheight, channels, bottomcount, pad_size_,
            rbot1, rbot2, bottom1_diff, top_diff);
        CORRELATION_CUDA_CHECK(cudaPeekAtLastError());
        }
    }
}
}  // namespace cuda
template<typename Dtype>
inline void CorrelationForward(const Tensor<gpu, 4, Dtype> &out,
                               const Tensor<gpu, 4, Dtype> &data1,
                               const Tensor<gpu, 4, Dtype> &data2,
                               const Tensor<gpu, 4, Dtype> &tmp1,
                               const Tensor<gpu, 4, Dtype> &tmp2,
                               int top_channels_, int top_height_,
                               int top_width_, int pad_size_, bool is_multiply,
                               int max_displacement_, int kernel_size_,
                               int neighborhood_grid_radius_, int neighborhood_grid_width_,
                               int kernel_radius_, int stride1_, int stride2_
                           ) {
  cudaStream_t stream = Stream<gpu>::GetStream(out.stream_);
  cudaStream_t stream_tmp1 = Stream<gpu>::GetStream(tmp1.stream_);
  cudaStream_t stream_tmp2 = Stream<gpu>::GetStream(tmp2.stream_);
  cuda::Forward_gpu(out, data1, data2, tmp1, tmp2, top_channels_, top_height_,
                    top_width_, pad_size_, is_multiply, max_displacement_, kernel_size_,
                    neighborhood_grid_radius_, neighborhood_grid_width_, kernel_radius_,
                    stride1_, stride2_, stream, stream_tmp1, stream_tmp2);
}

template<typename Dtype>
inline void CorrelationBackward(const Tensor<gpu, 4, Dtype> &out_grad,
                            const Tensor<gpu, 4, Dtype> &in_grad1,
                            const Tensor<gpu, 4, Dtype> &in_grad2,
                            const Tensor<gpu, 4, Dtype> &tmp1,
                            const Tensor<gpu, 4, Dtype> &tmp2,
                            int top_channels_, int top_height_,
                            int top_width_, int pad_size_, bool is_multiply,
                            int max_displacement_, int kernel_size_,
                            int neighborhood_grid_radius_, int neighborhood_grid_width_,
                            int  kernel_radius_, int stride1_,
                            int stride2_, int num, int channels, int height, int width
                            ) {
  cudaStream_t stream0 = Stream<gpu>::GetStream(in_grad1.stream_);
  cudaStream_t stream1 = Stream<gpu>::GetStream(in_grad2.stream_);
  cuda::Backward_gpu(out_grad, in_grad1, in_grad2, tmp1, tmp2, top_channels_,
                      top_height_, top_width_, pad_size_, is_multiply,
                      max_displacement_, kernel_size_, neighborhood_grid_radius_,
                      neighborhood_grid_width_, kernel_radius_, stride1_, stride2_,
                      stream0, stream1, num, channels, height, width);
}
}  // namespace mshadow
namespace mxnet {
namespace op {
template<>
Operator* CreateOp<gpu>(CorrelationParam param) {
  return new CorrelationOp<gpu>(param);
}
}  // namespace op
}  // namespace mxnet
