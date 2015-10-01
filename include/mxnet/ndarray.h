/*!
 *  Copyright (c) 2015 by Contributors
 * \file ndarray.h
 * \brief NDArray interface that handles array arithematics.
 */
#ifndef MXNET_NDARRAY_H_
#define MXNET_NDARRAY_H_

#include <dmlc/base.h>
#include <dmlc/logging.h>
#include <dmlc/io.h>
#include <dmlc/type_traits.h>
#include <dmlc/registry.h>
#include <vector>
#include <string>
#include <memory>
#include "./base.h"
#include "./storage.h"
#include "./engine.h"

// check c++11
#if DMLC_USE_CXX11 == 0
#error "cxx11 was required for ndarray module"
#endif

namespace mxnet {
/*!
 * \brief ndarray interface
 */
class NDArray {
 public:
  /*! \brief default cosntructor */
  NDArray() {}
  /*!
   * \brief constructing a new dynamic NDArray
   * \param shape the shape of array
   * \param ctx context of NDArray
   * \param delay_alloc whether delay the allocation
   */
  NDArray(const TShape &shape, Context ctx,
          bool delay_alloc = false)
      : ptr_(std::make_shared<Chunk>(shape.Size(), ctx, delay_alloc)), shape_(shape), offset_(0) {
  }
  /*!
   * \brief constructing a static NDArray that shares data with TBlob
   *  Use with caution: allocate ONLY ONE NDArray for each TBlob,
   *  make sure the memory region is available through out the life of NDArray
   * \param data the memory content of static data
   * \param dev_id the device id this tensor sits at
   */
  NDArray(const TBlob &data, int dev_id)
      : ptr_(std::make_shared<Chunk>(data, dev_id)), shape_(data.shape_), offset_(0) {
  }
  /*!
   * \return the shape of current NDArray
   */
  inline const TShape &shape() const {
    return shape_;
  }
  /*!
   * \return the data TBlob
   */
  inline TBlob data() const {
    return TBlob(static_cast<real_t*>(ptr_->shandle.dptr) + offset_, \
                 shape_, ptr_->shandle.ctx.dev_mask());
  }
  /*!
   * \return the context of NDArray, this function is only valid when the NDArray is not empty
   */
  inline Context ctx() const {
    return ptr_->shandle.ctx;
  }
  /*! \return whether this ndarray is not initialized */
  inline bool is_none() const {
    return ptr_.get() == nullptr;
  }
  /*!
   * \brief Block until all the pending write operations with respect
   *    to current NDArray are finished, and read can be performed.
   */
  inline void WaitToRead() const {
    if (is_none()) return;
    Engine::Get()->WaitForVar(ptr_->var);
  }
  /*!
   * \brief Block until all the pending read/write operations with respect
   *    to current NDArray are finished, and write can be performed.
   */
  inline void WaitToWrite() const {
    if (is_none()) return;
    /*!
     * Push an empty mutable function to flush all preceding reads to the
     * variable.
     */
    Engine::Get()->PushSync([](RunContext) {}, Context{}, {}, {ptr_->var});
    Engine::Get()->WaitForVar(ptr_->var);
  }
  /*! \return the associated variable of the ndarray.*/
  inline Engine::VarHandle var() const {
    return ptr_->var;
  }
  /*!
   * \brief save the content into binary stream
   * \param strm the output stream
   */
  void Save(dmlc::Stream *strm) const;
  /*!
   * \brief load the content from binary stream
   * \param strm the output stream
   * \return whether the load is successful
   */
  bool Load(dmlc::Stream *strm);
  /*!
   * \brief set all the elements in ndarray to be scalar
   * \param scalar the scalar to set
   * \return reference of self
   */
  NDArray &operator=(real_t scalar);
  /*!
   * \brief elementwise add to current space
   *  this mutate the current NDArray
   * \param src the data to add
   * \return reference of self
   */
  NDArray &operator+=(const NDArray &src);
  /*!
   * \brief elementwise add to current space
   *  this mutate the current NDArray
   * \param src the data to add
   * \return reference of self
   */
  NDArray &operator+=(const real_t &src);
  /*!
   * \brief elementwise subtract from current ndarray
   * this mutate the current NDArray
   * \param src the data to substract
   * \return reference of self
   */
  NDArray &operator-=(const NDArray &src);
  /*!
   * \brief elementwise subtract from current ndarray
   * this mutate the current NDArray
   * \param src the data to substract
   * \return reference of self
   */
  NDArray &operator-=(const real_t &src);
  /*!
   * \brief elementwise multiplication to current ndarray
   *  this mutate the current NDArray
   * \param src the data to substract
   * \return reference of self
   */
  NDArray &operator*=(const NDArray &src);
  /*!
   * \brief elementwise multiplication to current ndarray
   *  this mutate the current NDArray
   * \param src the data to substract
   * \return reference of self
   */
  NDArray &operator*=(const real_t &src);
  /*!
   * \brief elementwise division from current ndarray
   *  this mutate the current NDArray
   * \param src the data to substract
   * \return reference of self
   */
  NDArray &operator/=(const NDArray &src);
  /*!
   * \brief elementwise division from current ndarray
   *  this mutate the current NDArray
   * \param src the data to substract
   * \return reference of self
   */
  NDArray &operator/=(const real_t &src);
  /*!
   * \brief return transpose of current NDArray
   * \return a new transposed NDArray
   */
  NDArray T() const;
  /*!
   * \brief return a new copy this NDArray
   * \param ctx the new context of this NDArray
   * \return the new copy
   */
  NDArray Copy(Context ctx) const;
  /*!
   * \brief Do a synchronize copy from a continugous CPU memory region.
   *
   *  This function will call WaitToWrite before the copy is performed.
   *  This is useful to copy data from existing memory region that are
   *  not wrapped by NDArray(thus dependency not being tracked).
   *
   * \param data the data source to copy from.
   * \param size the memory size we want to copy from.
   */
  void SyncCopyFromCPU(const real_t *data, size_t size) const;
  /*!
   * \brief Do a synchronize copy to a continugous CPU memory region.
   *
   *  This function will call WaitToRead before the copy is performed.
   *  This is useful to copy data from existing memory region that are
   *  not wrapped by NDArray(thus dependency not being tracked).
   *
   * \param data the data source to copyinto.
   * \param size the memory size we want to copy into.
   */
  void SyncCopyToCPU(real_t *data, size_t size) const;
  /*!
   * \brief Slice a NDArray
   * \param begin begin index in first dim
   * \param end end index in first dim
   * \return sliced NDArray
   */
  inline NDArray Slice(index_t begin, index_t end) const {
    NDArray ret = *this;
    CHECK(!is_none()) << "NDArray is not initialized";
    CHECK_GE(shape_[0], end) << "Slice end index out of range";
    size_t length = 1;
    for (index_t i = 1; i < shape_.ndim(); ++i) {
      length *= shape_[i];
    }
    ret.offset_ += begin * length;
    ret.shape_[0] = end - begin;
    return ret;
  }
  /*!
   * \brief Get an reshaped NDArray
   * \param shape new shape
   * \return NDArray in new shape
   */
  inline NDArray Reshape(const TShape &shape) const {
    CHECK_GE(shape_.Size(), shape.Size())
        << "NDArray.Reshape: target shape size is different from current shape";
    NDArray ret = *this;
    ret.shape_ = shape;
    return ret;
  }
  /*!
   * \brief Allocate the space if it is delayed allocated.
   * This is an internal function used by system that normal user should not use
   */
  inline void CheckAndAlloc() const {
    ptr_->CheckAndAlloc();
  }
  /*!
   * \brief Save list of narray into the file.
   * \param fname name of the file.
   * \param data the NDArrays to be saved.
   * \param keys the name of the NDArray, optional, can be zero length.
   */
  static void Save(const std::string& fname,
                   const std::vector<NDArray>& data,
                   const std::vector<std::string>& names);
  /*!
   * \brief Load list of narray into from the file.
   * \param fname name of the file.
   * \param data the NDArrays to be loaded
   * \param keys the name of the NDArray, if saved in the file.
   */
  static void Load(const std::string& fname,
                   std::vector<NDArray>* data,
                   std::vector<std::string>* keys);

 private:
  /*! \brief the real data chunk that backs NDArray */
  struct Chunk {
    /*! \brief storage handlefrom storage engine */
    Storage::Handle shandle;
    /*! \brief variable from engine */
    Engine::VarHandle var;
    /*!
     * \brief if this is true, this means the data do not come
     * from Storage, and do not need to be freed
     */
    bool static_data;
    /*! \brief whether allocation is delayed */
    bool delay_alloc;
    /*! \brief default cosntructor */
    Chunk() : static_data(true), delay_alloc(false) {
      var  = Engine::Get()->NewVariable();
    }
    /*! \brief construct from static data */
    Chunk(const TBlob &data, int dev_id)
        : static_data(true),
          delay_alloc(false) {
      var = Engine::Get()->NewVariable();
      if (data.dev_mask_ == cpu::kDevMask) {
        shandle.ctx = Context::CPU();
      } else {
        CHECK_EQ(data.dev_mask_, gpu::kDevMask);
        shandle.ctx = Context::GPU(dev_id);
      }
      shandle.dptr = data.dptr_;
      shandle.size = data.shape_.Size() * sizeof(real_t);
    }
    /*! \brief construct a new chunk */
    Chunk(uint64_t size, Context ctx, bool delay_alloc_)
        : static_data(false), delay_alloc(true) {
      var = Engine::Get()->NewVariable();
      shandle.size = size * sizeof(real_t);
      shandle.ctx = ctx;
      if (!delay_alloc_) this->CheckAndAlloc();
    }
    /*! \brief check if delay alloc is on, do alloc if not yet done */
    inline void CheckAndAlloc(void) {
      if (delay_alloc) {
        shandle = Storage::Get()->Alloc(shandle.size, shandle.ctx);
        delay_alloc = false;
      }
    }
    /*! \brief destructor */
    ~Chunk() {
      if (static_data || delay_alloc) {
        Engine::Get()->DeleteVariable([](RunContext s) {}, shandle.ctx, var);
      } else {
        Storage::Handle h = this->shandle;
        Engine::Get()->DeleteVariable([h](RunContext s) {
            Storage::Get()->Free(h);
          }, shandle.ctx, var);
      }
    }
  };
  /*! \brief internal data of NDArray */
  std::shared_ptr<Chunk> ptr_;
  /*! \brief shape of current NDArray */
  TShape shape_;
  /*! \brief offset in chunk */
  size_t offset_;
};

/*!
 * \brief issue an copy operation from one NDArray to another
 *  the two ndarray can sit on different devices
 *  this operation will be scheduled by the engine
 *
 * \param from the ndarray we want to copy data from
 * \param to the target ndarray
 * \param priority Priority of the action.
 * \note The function name explicitly marks the order of from and to
 *     due to different possible convention carried by copy function.
 */
void CopyFromTo(const NDArray &from, NDArray *to, int priority = 0);

/*!
 * \brief Perform elementwise sum over each data from source, store result into out.
 * \param source the ndarray we want to sum
 * \param to the target ndarray
 * \param priority Priority of the action.
 */
void ElementwiseSum(const std::vector<NDArray> &source, NDArray *out, int priority = 0);

/*!
 * \brief elementwise add
 * \param lhs left operand
 * \param rhs right operand
 * \return a new result ndarray
 */
NDArray operator+(const NDArray &lhs, const NDArray &rhs);
/*!
 * \brief elementwise add
 * \param lhs left operand
 * \param rhs right operand
 * \return a new result ndarray
 */
NDArray operator+(const NDArray &lhs, const real_t &rhs);
/*!
 * \brief elementwise substraction
 * \param lhs left operand
 * \param rhs right operand
 * \return a new result ndarray
 */
NDArray operator-(const NDArray &lhs, const NDArray &rhs);
/*!
 * \brief elementwise substraction
 * \param lhs left operand
 * \param rhs right operand
 * \return a new result ndarray
 */
NDArray operator-(const NDArray &lhs, const real_t &rhs);
/*!
 * \brief elementwise multiplication
 * \param lhs left operand
 * \param rhs right operand
 * \return a new result ndarray
 */
NDArray operator*(const NDArray &lhs, const NDArray &rhs);\
/*!
 * \brief elementwise multiplication
 * \param lhs left operand
 * \param rhs right operand
 * \return a new result ndarray
 */
NDArray operator*(const NDArray &lhs, const real_t &rhs);
/*!
 * \brief elementwise division
 * \param lhs left operand
 * \param rhs right operand
 * \return a new result ndarray
 */
NDArray operator/(const NDArray &lhs, const NDArray &rhs);
/*!
 * \brief elementwise division
 * \param lhs left operand
 * \param rhs right operand
 * \return a new result ndarray
 */
NDArray operator/(const NDArray &lhs, const real_t &rhs);

/*!
 * \brief Seed the random number generator.
 * \param seed the seed to set to global random number generators.
 */
void RandomSeed(uint32_t seed);
/*!
 * \brief Sample uniform distribution for each elements of out.
 * \param begin lower bound of distribution.
 * \param end upper bound of distribution.
 * \param out output NDArray.
 */
void SampleUniform(real_t begin, real_t end, NDArray *out);

/*!
 * \brief Sample gaussian distribution for each elements of out.
 * \param mu mean of gaussian distribution.
 * \param sigma standard deviation of gaussian distribution.
 * \param out output NDArray.
 */
void SampleGaussian(real_t mu, real_t sigma, NDArray *out);
//--------------------------------------------------------------
// The following part are API Registration of NDArray functions.
//--------------------------------------------------------------
/*! \brief definition of NDArray function */
typedef std::function<void (NDArray **used_vars,
                            real_t *scalars,
                            NDArray **mutate_vars)> NDArrayAPIFunction;
/*! \brief mask information on how functions can be exposed */
enum NDArrayFunctionTypeMask {
  /*! \brief all the use_vars should go before scalar */
  kNDArrayArgBeforeScalar = 1,
  /*! \brief all the scalar should go before use_vars */
  kScalarArgBeforeNDArray = 1 << 1,
  /*!
   * \brief whether this function allows the handles in the target to
   *  be empty NDArray that are not yet initialized, and will initialize
   *  them when the function is invoked.
   *
   *  most function should support this, except copy between different
   *  devices, which requires the NDArray to be pre-initialized with context
   */
  kAcceptEmptyMutateTarget = 1 << 2
};
/*! \brief Registry entry for NDArrayFunction */
struct NDArrayFunctionReg
    : public dmlc::FunctionRegEntryBase<NDArrayFunctionReg,
                                        NDArrayAPIFunction> {
  /*! \brief number of variable used by this function */
  unsigned num_use_vars;
  /*! \brief number of variable mutated by this function */
  unsigned num_mutate_vars;
  /*! \brief number of scalars used by this function */
  unsigned num_scalars;
  /*! \brief information on how function should be called from API */
  int type_mask;
  /*!
   * \brief constructor
   */
  NDArrayFunctionReg()
      : num_use_vars(0),
        num_mutate_vars(0),
        num_scalars(0),
        type_mask(0) {}
  /*!
   * \brief set the function body to a NDArray setvalue function
   *  this will also auto set the parameters correctly
   * \param fsetvalue function body to set
   * \return ref to the registered entry, used to set properties
   */
  inline NDArrayFunctionReg &set_function(void (*fsetvalue)(const real_t &rhs,
                                                            NDArray *out)) {
    body = [fsetvalue] (NDArray **used_vars, real_t *s, NDArray **mutate_vars) {
      (*fsetvalue)(s[0], mutate_vars[0]);
    };
    num_mutate_vars = 1; num_scalars = 1;
    this->add_argument("src", "real_t", "Source input to the function.");
    return *this;
  }
  /*!
   * \brief set the function body to a binary NDArray function
   *  this will also auto set the parameters correctly
   * \param fbinary function body to set
   * \return ref to the registered entry, used to set properties
   */
  inline NDArrayFunctionReg &set_function(void (*fbinary)(const NDArray &lhs,
                                                          const NDArray &rhs,
                                                          NDArray *out)) {
    body = [fbinary] (NDArray **used_vars,
                      real_t *s, NDArray **mutate_vars) {
      (*fbinary)(*used_vars[0], *used_vars[1], mutate_vars[0]);
    };
    num_use_vars = 2; num_mutate_vars = 1;
    type_mask = kNDArrayArgBeforeScalar | kAcceptEmptyMutateTarget;
    this->add_argument("lhs", "NDArray", "Left operand to the function.");
    this->add_argument("rhs", "NDArray", "Right operand to the function.");
    return *this;
  }
  /*!
   * \brief set the function body to a binary NDArray function
   *  this will also auto set the parameters correctly
   * \param fscalar function body to set
   * \return ref to the registered entry, used to set properties
   */
  inline NDArrayFunctionReg &set_function(void (*fscalar)(const NDArray &lhs,
                                                          const real_t &rhs,
                                                          NDArray *out)) {
    body = [fscalar] (NDArray **used_vars,
                      real_t *s, NDArray **mutate_vars) {
      (*fscalar)(*used_vars[0], s[0], mutate_vars[0]);
    };
    num_use_vars = 1; num_mutate_vars = 1; num_scalars = 1;
    type_mask = kNDArrayArgBeforeScalar | kAcceptEmptyMutateTarget;
    this->add_argument("lhs", "NDArray", "Left operand to the function.");
    this->add_argument("rhs", "real_t", "Right operand to the function.");
    return *this;
  }
  /*!
   * \brief set the function body to a unary NDArray function
   *  this will also auto set the parameters correctly
   * \param funary function body to set
   * \return ref to the registered entry, used to set properties
   */
  inline NDArrayFunctionReg &set_function(void (*funary)(const NDArray &src,
                                                         NDArray *out)) {
    body = [funary] (NDArray **used_vars,
                     real_t *s, NDArray **mutate_vars) {
      (*funary)(*used_vars[0], mutate_vars[0]);
    };
    num_use_vars = 1; num_mutate_vars = 1;
    type_mask = kNDArrayArgBeforeScalar | kAcceptEmptyMutateTarget;
    this->add_argument("src", "NDArray", "Source input to the function.");
    return *this;
  }
  /*!
   * \brief set the number of mutate variables
   * \param n number of mutate variablesx
   * \return ref to the registered entry, used to set properties
   */
  inline NDArrayFunctionReg &set_num_use_vars(unsigned n) {
    num_use_vars = n; return *this;
  }
  /*!
   * \brief set the number of mutate variables
   * \param n number of mutate variablesx
   * \return ref to the registered entry, used to set properties
   */
  inline NDArrayFunctionReg &set_num_mutate_vars(unsigned n) {
    num_mutate_vars = n; return *this;
  }
  /*!
   * \brief set the number of scalar arguments
   * \param n number of scalar arguments
   * \return ref to the registered entry, used to set properties
   */
  inline NDArrayFunctionReg &set_num_scalars(unsigned n) {
    num_scalars = n; return *this;
  }
  /*!
   * \brief set type mask
   * \param tmask typemask
   * \return ref to the registered entry, used to set properties
   */
  inline NDArrayFunctionReg &set_type_mask(int tmask) {
    type_mask = tmask; return *this;
  }
};  // NDArrayFunctionReg

/*!
 * \brief Macro to register NDArray function
 *
 * Example: the following code is example to register a plus
 * \code
 *
 * REGISTER_NDARRAY_FUN(Plus)
 * .set_function(Plus);
 *
 * \endcode
 */
#define MXNET_REGISTER_NDARRAY_FUN(name)                                 \
  DMLC_REGISTRY_REGISTER(::mxnet::NDArrayFunctionReg, NDArrayFunctionReg, name)

}  // namespace mxnet

namespace dmlc {
/*!\brief traits */
DMLC_DECLARE_TRAITS(has_saveload, mxnet::NDArray, true);
}  // namespace dmlc
#endif  // MXNET_NDARRAY_H_
