/*!
 *  Copyright (c) 2015 by Contributors
 * \file inst_vector.h
 * \brief holder of a sequence of DataInst in CPU
 *        that are not necessarily of same shape
 */

#ifndef MXNET_IO_INST_VECTOR_H_
#define MXNET_IO_INST_VECTOR_H_

#include <mxnet/io.h>
#include <mxnet/base.h>
#include <dmlc/base.h>
#include <mshadow/tensor.h>
#include <vector>
#include <string>

namespace mxnet {
namespace io {
/*!
 * \brief a vector of tensor with various shape
 *
 * data are stored in memory continuously
 */
template<int dim, typename DType>
class TensorVector {
 public:
  TensorVector(void) {
    this->Clear();
  }
  /*! \brief get the buffer to the i-th tensor */
  inline mshadow::Tensor<cpu, dim, DType>
  operator[](size_t i) const {
    CHECK_LT(i + 1, offset_.size());
    CHECK_EQ(shape_[i].Size(), offset_[i + 1] - offset_[i]);
    return mshadow::Tensor<cpu, dim, DType>
        ((DType*)dmlc::BeginPtr(content_) + offset_[i], shape_[i]);  // NOLINT(*)
  }
  inline mshadow::Tensor<cpu, dim, DType> Back() const {
    return (*this)[Size() - 1];
  }
  inline size_t Size(void) const {
    return shape_.size();
  }
  /*! \brief allocate space given the shape (data are copied) */
  inline void Push(mshadow::Shape<dim> shape) {
    shape_.push_back(shape);
    offset_.push_back(offset_.back() + shape.Size());
    content_.resize(offset_.back());
  }
  inline void Clear(void) {
    offset_.clear();
    offset_.push_back(0);
    content_.clear();
    shape_.clear();
  }

 private:
  // offset of the data content
  std::vector<size_t> offset_;
  // data content
  std::vector<DType> content_;
  // shape of data
  std::vector<mshadow::Shape<dim> > shape_;
};

/*!
 * \brief a list of (label, example) pairs, examples can have various shape
 */
template<typename DType = real_t>
class InstVector {
 public:
  /*! \brief return the number of (label, example) pairs */
  inline size_t Size(void) const {
    return index_.size();
  }
  // get index
  inline unsigned Index(unsigned i) const {
    return index_[i];
  }
  // instance
  /* \brief get the i-th (label, example) pair */
  inline DataInst operator[](size_t i) const {
    DataInst inst;
    inst.index = index_[i];
    inst.data.push_back(TBlob(data_[i]));
    inst.data.push_back(TBlob(label_[i]));
    return inst;
  }
  /* \brief get the last (label, example) pair */
  inline DataInst Back() const {
    return (*this)[Size() - 1];
  }
  inline void Clear(void) {
    index_.clear();
    data_.Clear();
    label_.Clear();
  }
  /*
   * \brief push a (label, example) pair
   * only reserved the space, while the data is not copied
   */
  inline void Push(unsigned index,
                   mshadow::Shape<3> dshape,
                   mshadow::Shape<1> lshape) {
    index_.push_back(index);
    data_.Push(dshape);
    label_.Push(lshape);
  }
  /*! \return the data content */
  inline const TensorVector<3, DType>& data() const {
    return data_;
  }
  /*! \return the label content */
  inline const TensorVector<1, real_t>& label() const {
    return label_;
  }

 private:
  /*! \brief index of the data */
  std::vector<unsigned> index_;
  // label
  TensorVector<3, DType> data_;
  // data
  TensorVector<1, real_t> label_;
};

/*!
 * \brief tblob batch
 *
 * data are stored in tblob before going into NDArray
 */
struct TBlobBatch {
 public:
  /*! \brief unique id for instance, can be NULL, sometimes is useful */
  unsigned *inst_index;
  /*! \brief number of instance */
  mshadow::index_t batch_size;
  /*! \brief number of padding elements in this batch,
       this is used to indicate the last elements in the batch are only padded up to match the batch, and should be discarded */
  mshadow::index_t num_batch_padd;
  /*! \brief content of dense data */
  std::vector<TBlob> data;
  /*! \brief extra data to be fed to the network */
  std::string extra_data;
  /*! \brief constructor */
  TBlobBatch(void) {
    inst_index = NULL;
    batch_size = 0; num_batch_padd = 0;
  }
  /*! \brief destructor */
  ~TBlobBatch() {
    delete inst_index;
  }
};  // struct TBlobBatch

class TBlobContainer : public mshadow::TBlob {
 public:
  TBlobContainer(void)
    : mshadow::TBlob(), tensor_container_(nullptr) {}
  ~TBlobContainer() {
    if (tensor_container_) {
      release();
    }
  }
  void resize(const mshadow::TShape &shape, int type_flag) {
    if (tensor_container_) {
      CHECK_EQ(this->type_flag_, type_flag);
      this->shape_ = shape;
      resize();
    } else {
      this->type_flag_ = type_flag;
      this->shape_ = shape;
      create();
    }
    this->stride_ = shape_[shape_.ndim() - 1];
  }

 private:
  void create() {
    CHECK(tensor_container_ == nullptr);
    CHECK_EQ(this->dev_mask_, mshadow::cpu::kDevMask);
    MSHADOW_TYPE_SWITCH(this->type_flag_, DType, {
        auto tensor_container = new mshadow::TensorContainer<mshadow::cpu, 1, DType>(false);
        tensor_container->Resize(mshadow::Shape1(shape_.Size()));
        dptr_ = tensor_container->dptr_;
        tensor_container_ = tensor_container;
    });
  }
  void resize() {
    MSHADOW_TYPE_SWITCH(this->type_flag_, DType, {
        auto tensor_container =
          (mshadow::TensorContainer<mshadow::cpu, 1, DType>*) tensor_container_;
        tensor_container->Resize(mshadow::Shape1(shape_.Size()));
    });
  }
  void release() {
    MSHADOW_TYPE_SWITCH(this->type_flag_, DType, {
        auto tensor_container =
          (mshadow::TensorContainer<mshadow::cpu, 1, DType>*) tensor_container_;
        delete tensor_container;
    });
  }

  void* tensor_container_;
};

}  // namespace io
}  // namespace mxnet
#endif  // MXNET_IO_INST_VECTOR_H_
