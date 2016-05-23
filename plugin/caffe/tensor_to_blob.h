#ifndef CAFFE_TENSOR_TO_BLOB_HPP_ 
#define CAFFE_TENSOR_TO_BLOB_HPP_

#include<caffe/blob.hpp>
#include<mshadow/tensor.h>
namespace caffe{

// Declare Memory Type for Caffe blob
namespace caffememtype {
enum caffeMemoryTypes {Data, Grad, Non};
}  // caffememtype

template<typename Device, int dimension>
void SetDataGradToBlob(Blob<float> *blob,
                       caffememtype::caffeMemoryTypes memType,
                       ::mshadow::Tensor<Device, dimension> *tensor);

template<typename Device, int dimension>
void TensorToBlob(Blob<float> *blob,
                  caffememtype::caffeMemoryTypes memType0,
                  ::mshadow::Tensor<Device, dimension> *tensor0, 
                  caffememtype::caffeMemoryTypes memType1 = caffememtype::Non, 
                  ::mshadow::Tensor<Device, dimension> *tensor1 = NULL){
  std::vector<int> shape;
  shape.resize(dimension);

  //In Caffe's blob, shape[0] is batch size. shape[1] is length of dimension
  
#if defined(CAFFE_DEBUG)
  std::cout << "Caffe-0 is batch size Caffe-1 is length" << std::endl;
  std::cout << "Tensor To Blob" << std::endl;
  for(int i = 0; i < dimension; ++ i)
    std::cout << i << " " << tensor0->shape_[i] << std::endl;
#endif

  for(int i = 0; i < dimension; ++ i){
    shape[i] = tensor0->shape_[i];
    if(tensor1 != NULL)
      CHECK_EQ(tensor0->shape_[i], tensor1->shape_[i]);
  }
 
  blob->Reshape(shape);
  SetDataGradToBlob<Device, dimension>(blob, memType0, tensor0);
  if((memType1 != caffememtype::Non)&&(tensor1 != NULL))
    SetDataGradToBlob<Device, dimension>(blob, memType1, tensor1);
}

}

#endif
