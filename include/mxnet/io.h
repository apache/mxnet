/*!
 *  Copyright (c) 2015 by Contributors
 * \file io.h
 * \brief mxnet io data structure and data iterator
 */
#ifndef MXNET_IO_H_
#define MXNET_IO_H_
#include <dmlc/data.h>
#include <dmlc/registry.h>
#include <vector>
#include <string>
#include <utility>
#include "./base.h"

namespace mxnet {
/*!
 * \brief iterator type
 * \tparam DType data type
 */
template<typename DType>
class IIterator : public dmlc::DataIter<DType> {
 public:
  /*!
   * \brief set the parameters and init iter
   * \param kwargs key-value pairs
   */ 
  virtual void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) = 0;
  /*! \brief reset the iterator */
  virtual void BeforeFirst(void) = 0;
  /*! \brief move to next item */
  virtual bool Next(void) = 0;
  /*! \brief get current data */
  virtual const DType &Value(void) const = 0;
  /*! \brief constructor */
  virtual ~IIterator(void) {}
  /*! \brief store the name of each data, it could be used for making NArrays */
  std::vector<std::string> data_names;
  /*! \brief set data name to each attribute of data */
  inline void SetDataName(const std::string data_name){
    data_names.push_back(data_name);
  }
};  // class IIterator

/*! \brief a single data instance */
struct DataInst {
  /*! \brief unique id for instance */
  unsigned index;
  /*! \brief content of data */
  std::vector<TBlob> data;
  /*! \brief extra data to be fed to the network */
  std::string extra_data;
};  // struct DataInst

/*!
 * \brief a standard batch of data commonly used by iterator
 *      a databatch contains multiple TBlobs. Each Tblobs has
 *      a name stored in a map. There's no different between
 *      data and label, how we use them is to see the DNN implementation.
 */
struct DataBatch {
 public:
  /*! \brief unique id for instance, can be NULL, sometimes is useful */
  unsigned *inst_index;
  /*! \brief number of instance */
  mshadow::index_t batch_size;
  /*! \brief number of padding elements in this batch,
       this is used to indicate the last elements in the batch are only padded up to match the batch, and should be discarded */
  mshadow::index_t num_batch_padd;
 public:
  /*! \brief content of dense data, if this DataBatch is dense */
  std::vector<TBlob> data;
  /*! \brief extra data to be fed to the network */
  std::string extra_data;
 public:
  /*! \brief constructor */
  DataBatch(void) {
    inst_index = NULL;
    batch_size = 0; num_batch_padd = 0;
  }
  /*! \brief giving name to the data */
  void Naming(std::vector<std::string> names);
};  // struct DataBatch

/*! \brief typedef the factory function of data iterator */
typedef IIterator<DataBatch> *(*DataIteratorFactory)();
/*!
 * \brief Registry entry for DataIterator factory functions.
 */
struct DataIteratorReg
    : public dmlc::FunctionRegEntryBase<DataIteratorReg,
                                        DataIteratorFactory> {
};
//--------------------------------------------------------------
// The following part are API Registration of Iterators
//--------------------------------------------------------------
/*!
 * \brief Macro to register Iterators
 *
 * \code
 * // example of registering a mnist iterator
 * REGISTER_IO_ITERATOR(MNIST, MNISTIterator)
 * .describe("Mnist data iterator");
 *
 * \endcode
 */
#define MXNET_REGISTER_IO_ITER(name, DataIteratorType)          \
  static ::mxnet::IIterator<DataBatch>* __create__ ## DataIteratorType ## __() { \
    return new DataIteratorType;                                    \
  }                                                                     \
  DMLC_REGISTRY_REGISTER(::mxnet::DataIteratorReg, DataIteratorReg, name) \
  .set_body(__create__ ## DataIteratorType ## __)
/*!
 * \brief Macro to register chained Iterators
 *
 * \code
 * // example of registering a imagerec iterator
 * MXNET_REGISTER_IO_CHAINED_ITERATOR(ImageRec, ImageRecordIter, BatchIter)
 * .describe("batched image record data iterator");
 *
 * \endcode
 */
#define MXNET_REGISTER_IO_CHAINED_ITER(name, ChainedDataIterType, HoldingDataIterType)          \
  static ::mxnet::IIterator<DataBatch>* __create__ ## ChainedDataIteratorType ## __() { \
    return new HoldingDataIterType(new ChainedDataIterType);                                    \
  }                                                                     \
  DMLC_REGISTRY_REGISTER(::mxnet::DataIteratorReg, DataIteratorReg, name) \
  .set_body(__create__ ## ChainedDataIteratorType ## __)
}  // namespace mxnet
#endif  // MXNET_IO_H_
