/*!
 * Copyright (c) 2016 by Contributors
 * \file postprocess_rpn.cu
 * \brief post process of rpn operator
 * \author Ming Zhang
*/
#include "./postprocess_rpn-inl.h"
#include "./mshadow_op.h"


namespace mshadow {

namespace cuda {

__global__ void PostProcessRPNForwardKernel2(
                int count,
                const float *pfCls, const float *pfReg, 
                const float *pfAnchor, const float *pfOtherinfo, 
                int dwAnchorNum, int dwFeatH, int dwFeatW, 
                float *pfBBs, int dwMaxBBNum, int *pdwbb_num_now) {
#if 1     
  float clsthreshold = pfOtherinfo[0];
  int originalH = pfOtherinfo[1];
  int originalW = pfOtherinfo[2]; 
//  printf("clsthreshold:%.1f, originalH:%d, originalW:%d\n", clsthreshold, originalH, originalW);
//  __syncthreads();
  int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;

  if (1 && index < count)
  {
    int nownum = *pdwbb_num_now;
//    printf("bidxx:%d-bidxy:%d-gdimx:%d-bdimx:%d-tidxx:%d, index:%d, nownum:%d\n", blockIdx.x, blockIdx.y, gridDim.x, blockDim.x, threadIdx.x, index, nownum);
//    printf("bidxx:%d-bidxy:%d-gdimx:%d-bdimx:%d-tidxx:%d, index:%d\n", blockIdx.x, blockIdx.y, gridDim.x, blockDim.x, threadIdx.x, index);
//    __syncthreads();
#if 1
    if (nownum >= 0 && nownum < dwMaxBBNum)
    {
      int dwFeatSize = dwFeatH * dwFeatW;
      int dwAnchorI = index / dwFeatSize;
      int dwRI = (index - dwAnchorI * dwFeatSize) / dwFeatW;
      int dwCI = (index - dwAnchorI * dwFeatSize) % dwFeatW;
      int dwOft = dwRI * dwFeatW + dwCI;
      int dwAnchorOft = dwAnchorI * dwFeatSize;
      const float *pfNowAnchor = pfAnchor + dwAnchorI * 2;
      if (pfCls[dwOft + dwAnchorOft] > clsthreshold)
      {
        float fCY = pfReg[dwAnchorOft * 4 + 0 * dwFeatSize + dwOft];
        float fCX = pfReg[dwAnchorOft * 4 + 1 * dwFeatSize + dwOft];
        float fH = pfReg[dwAnchorOft * 4 + 2 * dwFeatSize + dwOft];
        float fW = pfReg[dwAnchorOft * 4 + 3 * dwFeatSize + dwOft];
        fCY = fCY * pfNowAnchor[0] + ((float)(dwRI) * originalH) / dwFeatH;
        fCX = fCX * pfNowAnchor[1] + ((float)(dwCI) * originalW) / dwFeatW;
        fH = expf(fH) * pfNowAnchor[0];
        fW = expf(fW) * pfNowAnchor[1];
        atomicInc((unsigned int*)pdwbb_num_now, dwMaxBBNum);
        nownum = *pdwbb_num_now;
       
        if (nownum > 0)
        {
          pfBBs[(nownum-1) * 4 + 0] = fCY;
          pfBBs[(nownum-1) * 4 + 1] = fCX;
          pfBBs[(nownum-1) * 4 + 2] = fH;
          pfBBs[(nownum-1) * 4 + 3] = fW;

//          printf("bidxx:%d-bidxy:%d-gdimx:%d-bdimx:%d-tidxx:%d, index:%d, nownum:%d\n", blockIdx.x, blockIdx.y, gridDim.x, blockDim.x, threadIdx.x, index, nownum);
//          __syncthreads();
        }
      }
    }
#endif
  }
#endif
}


inline void PostProcessRPNForward2(const Tensor<gpu, 4> &datacls_in,
                           const Tensor<gpu, 4> &datareg_in,
                           const Tensor<gpu, 2> &anchorinfo_in,
                           const Tensor<gpu, 1> &otherinfo_in,
                           Tensor<gpu, 3> &bb_out) {
  CHECK_EQ(datacls_in.size(0), datareg_in.size(0));

  int dwBatchNum = datacls_in.size(0);
  int dwAnchorNum = anchorinfo_in.size(0);
  int bb_maxnum_per_batch = bb_out.size(1);
  
  int dwFeatH = datacls_in.size(2);
  int dwFeatW = datacls_in.size(3);
  int dwBBMemLen = bb_out.MSize();
  cudaMemset(bb_out.dptr_, 0, dwBBMemLen*sizeof(float));
  int *pdwCounter = 0;
  cudaMalloc(&pdwCounter, dwBatchNum*sizeof(int));
  cudaMemset(pdwCounter, 0, dwBatchNum*sizeof(int));
//  printf("dwBBMemLen:%d\n", dwBBMemLen);
//  float *pfAnchorData = anchorinfo_in.dptr_;
//  printf("anchor[%dx%d]_address:%x\n", anchorinfo_in.size(0), anchorinfo_in.size(1), pfAnchorData);
//  for (int dwI = 0; dwI < 1; dwI++)
//  {
//    printf("anchor_%d:%.1f, %.1f\n", dwI, pfAnchorData[dwI * 2 + 0], pfAnchorData[dwI * 2 + 1]);
//  }
//  __syncthreads();
  
  int count = dwFeatH * dwFeatW * dwAnchorNum;
#if 1
  const int gridSize = (count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
  dim3 dimGrid(kMaxGridNum, (gridSize + kMaxGridNum - 1) / kMaxGridNum);
  dim3 dimBlock(kMaxThreadsPerBlock);
#else
  dim3 dimGrid(2, 2);
  dim3 dimBlock(2);
#endif
  CheckLaunchParam(dimGrid, dimBlock, "PostProcessRPN Forward");
  cudaStream_t stream = Stream<gpu>::GetStream(bb_out.stream_);
  
  for (int bi = 0; bi < dwBatchNum; bi++) {
//    printf("fucking start...(%d)\n", bi);
    const Tensor<gpu, 3> &datacls_onebatch = datacls_in[bi];
    const Tensor<gpu, 3> &datareg_onebatch = datareg_in[bi];
    Tensor<gpu, 2> bb_onebatch = bb_out[bi];
//    printf("count:%d\n", count);
//    printf("datacls_onebatch.dptr_:%x\n", datacls_onebatch.dptr_);
//    printf("datareg_onebatch.dptr_:%x\n", datareg_onebatch.dptr_);
//    printf("anchorinfo_in.dptr_:%x\n", anchorinfo_in.dptr_);
//    printf("otherinfo_in.dptr_:%x\n", otherinfo_in.dptr_);
//    printf("dwAnchorNum:%d, dwFeatH:%d, dwFeatW:%d\n", dwAnchorNum, dwFeatH, dwFeatW);
//    printf("bb_onebatch.dptr_:%x\n", bb_onebatch.dptr_);
//    printf("bb_maxnum_per_batch:%d\n", bb_maxnum_per_batch); 
#if 1   
    PostProcessRPNForwardKernel2<<<dimGrid, dimBlock, 0, stream>>>(
            count, 
            datacls_onebatch.dptr_, datareg_onebatch.dptr_, 
            anchorinfo_in.dptr_, otherinfo_in.dptr_, dwAnchorNum, dwFeatH, dwFeatW, bb_onebatch.dptr_, bb_maxnum_per_batch, pdwCounter+bi);
//    cudaThreadSynchronize();
#else
    PostProcessRPNForwardKernel1<<<dimGrid, dimBlock>>>();
#endif
//    printf("fucking ending...(%d)\n", bi);
  }
  cudaFree(pdwCounter);
}


__global__ void PostProcessRPNForwardKernel(
                int count,
                const float *pfClsAll, const float *pfRegAll, 
                const float *pfAnchor, const float *pfOtherinfo, 
                int dwBatchNum, int dwAnchorNum, int dwFeatH, int dwFeatW, 
                float *pfBBsAll, int dwMaxBBNum, int *pdwbb_num_all) {
#if 1     
  float clsthreshold = pfOtherinfo[0];
  int originalH = pfOtherinfo[1];
  int originalW = pfOtherinfo[2]; 
//  printf("clsthreshold:%.1f, originalH:%d, originalW:%d\n", clsthreshold, originalH, originalW);
//  __syncthreads();
  int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;

  if (1 && index < count)
  {
    int dwFeatSize = dwFeatH * dwFeatW;
    int dwFeatAnchorSize = dwFeatSize * dwAnchorNum;
    int dwBatchI = index / dwFeatAnchorSize;
    int dwAnchorI = (index - dwBatchI * dwFeatAnchorSize) / dwFeatSize;
    int dwRI = (index - dwBatchI * dwFeatAnchorSize - dwFeatSize * dwAnchorI) / dwFeatW;
    int dwCI = (index - dwBatchI * dwFeatAnchorSize - dwFeatSize * dwAnchorI) % dwFeatW;
    int dwOft = dwRI * dwFeatW + dwCI;
    int dwAnchorOft = dwAnchorI * dwFeatSize;
    const float *pfNowAnchor = pfAnchor + dwAnchorI * 2;
    const float *pfReg = pfRegAll + dwFeatAnchorSize * 4 * dwBatchI;
    const float *pfCls = pfClsAll + dwFeatAnchorSize * dwBatchI;
    float *pfBBs = pfBBsAll + dwBatchI * dwMaxBBNum * 5;
    int *pdwbb_num_now = pdwbb_num_all + dwBatchI;

//    printf("bidxx:%d-bidxy:%d-gdimx:%d-bdimx:%d-tidxx:%d, index:%d, nownum:%d\n", blockIdx.x, blockIdx.y, gridDim.x, blockDim.x, threadIdx.x, index, nownum);
//    printf("bidxx:%d-bidxy:%d-gdimx:%d-bdimx:%d-tidxx:%d, index:%d\n", blockIdx.x, blockIdx.y, gridDim.x, blockDim.x, threadIdx.x, index);
//    __syncthreads();
#if 1
    float fScore = pfCls[dwOft + dwAnchorOft];
    if (fScore > clsthreshold)
    {
      int nownum = atomicInc((unsigned int*)(pdwbb_num_now), dwMaxBBNum);
      if (nownum < dwMaxBBNum)
      {
        float fCY = pfReg[dwAnchorOft * 4 + 0 * dwFeatSize + dwOft];
        float fCX = pfReg[dwAnchorOft * 4 + 1 * dwFeatSize + dwOft];
        float fH = pfReg[dwAnchorOft * 4 + 2 * dwFeatSize + dwOft];
        float fW = pfReg[dwAnchorOft * 4 + 3 * dwFeatSize + dwOft];
        fCY = fCY * pfNowAnchor[0] + ((float)(dwRI) * originalH) / dwFeatH;
        fCX = fCX * pfNowAnchor[1] + ((float)(dwCI) * originalW) / dwFeatW;
        fH = expf(fH) * pfNowAnchor[0];
        fW = expf(fW) * pfNowAnchor[1];
       
        {
#if 0
          pfBBs[(nownum-1) * 5 + 0] = fScore;
          pfBBs[(nownum-1) * 5 + 1] = fCY;
          pfBBs[(nownum-1) * 5 + 2] = fCX;
          pfBBs[(nownum-1) * 5 + 3] = fH;
          pfBBs[(nownum-1) * 5 + 4] = fW;
#else
          pfBBs[nownum * 5 + 0] = fScore;
          pfBBs[nownum * 5 + 1] = fCY;
          pfBBs[nownum * 5 + 2] = fCX;
          pfBBs[nownum * 5 + 3] = fH;
          pfBBs[nownum * 5 + 4] = fW;

#endif
#if 0
          pfBBs[0] = fCY;
          pfBBs[1] = fCX;
          pfBBs[2] = fH;
          pfBBs[3] = fW;
#endif
//          printf("bidxx:%d-bidxy:%d-gdimx:%d-bdimx:%d-tidxx:%d, index:%d, nownum:%d\n", blockIdx.x, blockIdx.y, gridDim.x, blockDim.x, threadIdx.x, index, nownum);
//          __syncthreads();
        }
      }
    }
#endif
  }
#endif
}


inline void PostProcessRPNForward(const Tensor<gpu, 4> &datacls_in,
                           const Tensor<gpu, 4> &datareg_in,
                           const Tensor<gpu, 2> &anchorinfo_in,
                           const Tensor<gpu, 1> &otherinfo_in,
                           Tensor<gpu, 3> &bb_out) {
  CHECK_EQ(datacls_in.size(0), datareg_in.size(0));

  int dwBatchNum = datacls_in.size(0);
  int dwAnchorNum = anchorinfo_in.size(0);
  int bb_maxnum_per_batch = bb_out.size(1);
  
  int dwFeatH = datacls_in.size(2);
  int dwFeatW = datacls_in.size(3);
  int dwBBMemLen = bb_out.MSize();
  cudaMemset(bb_out.dptr_, 0, dwBBMemLen*sizeof(float));
  int *pdwCounter = 0;
  cudaMalloc(&pdwCounter, dwBatchNum*sizeof(int));
  cudaMemset(pdwCounter, 0, dwBatchNum*sizeof(int));
//  int *pdwCounterHost = (int*)malloc(sizeof(int)*dwBatchNum);
//  for (int dwI = 0; dwI < dwBatchNum; dwI++)
//  {
//    pdwCounterHost[dwI] = 1;
//  }
//  cudaMemcpy(pdwCounter, pdwCounterHost, dwBatchNum*sizeof(int), cudaMemcpyHostToDevice);
  
  int count = dwFeatH * dwFeatW * dwAnchorNum * dwBatchNum;
#if 1
  const int gridSize = (count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
  dim3 dimGrid(kMaxGridNum, (gridSize + kMaxGridNum - 1) / kMaxGridNum);
  dim3 dimBlock(kMaxThreadsPerBlock);
#else
  dim3 dimGrid(2, 2);
  dim3 dimBlock(2);
#endif
  CheckLaunchParam(dimGrid, dimBlock, "PostProcessRPN Forward");
  cudaStream_t stream = Stream<gpu>::GetStream(bb_out.stream_);
  
  PostProcessRPNForwardKernel<<<dimGrid, dimBlock, 0, stream>>>(
            count, 
            datacls_in.dptr_, datareg_in.dptr_, 
            anchorinfo_in.dptr_, otherinfo_in.dptr_, dwBatchNum, dwAnchorNum, dwFeatH, dwFeatW, 
            bb_out.dptr_, bb_maxnum_per_batch, pdwCounter);

#if 0
  {
    cudaMemcpy(pdwCounterHost, pdwCounter, dwBatchNum*sizeof(int), cudaMemcpyDeviceToHost);
    printf("mxnet=>dwCounter:");
    for (int i = 0; i < dwBatchNum; i++)
    {
      printf("%d:%d, ", i, pdwCounterHost[i]);
    }
    printf("\n");
    printf("mxnet=>bb_maxnum_per_batch:%d\n", bb_maxnum_per_batch);
    float *pfBBsAll = (float*)malloc(dwBBMemLen*sizeof(float));
    cudaMemcpy(pfBBsAll, bb_out.dptr_, dwBBMemLen*sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < dwBatchNum; i++)
    {
      float *pfNow = pfBBsAll + i * bb_maxnum_per_batch * 4;
      if (pdwCounterHost[i] > 0 || 1)
      {
        printf("mxnet==>%d, 0:%.2f, %.2f, %.2f, %.2f\n", i, pfNow[0], pfNow[1], pfNow[2], pfNow[3]);
      }
    }
    free(pfBBsAll);
  }
#endif
//  free(pdwCounterHost);
  cudaFree(pdwCounter);
}
  
} // namespace cuda

inline void PostProcessRPNForward(const Tensor<gpu, 4> &datacls_in,
                           const Tensor<gpu, 4> &datareg_in,
                           const Tensor<gpu, 2> &anchorinfo_in,
                           const Tensor<gpu, 1> &otherinfo_in,
                           Tensor<gpu, 3> &bb_out) {
//  printf("originalW:%d\n", originalW);                           
  cuda::PostProcessRPNForward(datacls_in, datareg_in, anchorinfo_in, otherinfo_in, bb_out);
//  cuda::PostProcessRPNForward2(datacls_in, datareg_in, anchorinfo_in, otherinfo_in, bb_out);
}

} // namespace mshadow


namespace mxnet {
namespace op {

template<>
Operator *CreateOp<gpu>(PostProcessRPNParam param) {
  return new PostProcessRPNOp<gpu>(param);
}

}  // namespace op
}  // namespace mxnet
