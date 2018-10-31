/*!
 *  Copyright (c) 2015 by Contributors
 * \file data.h
 * \brief defines common input data structure,
 *  and interface for handling the input data
 */
#ifndef DMLC_DATA_H_
#define DMLC_DATA_H_

#include <string>
#include <vector>
#include <map>
#include "./base.h"
#include "./io.h"
#include "./logging.h"
#include "./registry.h"

// To help C Preprocessor with processing c++ templated types
#define __DMLC_COMMA ,

namespace dmlc {
/*!
 * \brief this defines the float point
 * that will be used to store feature values
 */
typedef float real_t;

/*!
 * \brief this defines the unsigned integer type
 * that can normally be used to store feature index
 */
typedef unsigned index_t;

// This file describes common data structure that can be used
// for large-scale machine learning, this may not be a complete list
// But we will keep the most common and useful ones, and keep adding new ones
/*!
 * \brief data iterator interface
 *  this is not a C++ style iterator, but nice for data pulling:)
 *  This interface is used to pull in the data
 *  The system can do some useful tricks for you like pre-fetching
 *  from disk and pre-computation.
 *
 * Usage example:
 * \code
 *
 *   itr->BeforeFirst();
 *   while (itr->Next()) {
 *      const DType &batch = itr->Value();
 *      // some computations
 *   }
 * \endcode
 * \tparam DType the data type
 */
template<typename DType>
class DataIter {
 public:
  /*! \brief destructor */
  virtual ~DataIter(void) {}
  /*! \brief set before first of the item */
  virtual void BeforeFirst(void) = 0;
  /*! \brief move to next item */
  virtual bool Next(void) = 0;
  /*! \brief get current data */
  virtual const DType &Value(void) const = 0;
};

/*!
 * \brief one row of training instance
 * \tparam IndexType type of index
 * \tparam DType type of data (both label and value will be of DType
 */
template<typename IndexType, typename DType = real_t>
class Row {
 public:
  /*! \brief label of the instance */
  const DType *label;
  /*! \brief weight of the instance */
  const real_t *weight;
  /*! \brief session-id of the instance */
  const uint64_t *qid;
  /*! \brief length of the sparse vector */
  size_t length;
  /*!
   * \brief field of each instance
   */
  const IndexType *field;
  /*!
   * \brief index of each instance
   */
  const IndexType *index;
  /*!
   * \brief array value of each instance, this can be NULL
   *  indicating every value is set to be 1
   */
  const DType *value;
  /*!
   * \param i the input index
   * \return field for i-th feature
   */
  inline IndexType get_field(size_t i) const {
    return field[i];
  }
  /*!
   * \param i the input index
   * \return i-th feature
   */
  inline IndexType get_index(size_t i) const {
    return index[i];
  }
  /*!
   * \param i the input index
   * \return i-th feature value, this function is always
   *  safe even when value == NULL
   */
  inline DType get_value(size_t i) const {
    return value == NULL ? DType(1.0f) : value[i];
  }
  /*!
   * \return the label of the instance
   */
  inline DType get_label() const {
    return *label;
  }
  /*!
   * \return the weight of the instance, this function is always
   *  safe even when weight == NULL
   */
  inline real_t get_weight() const {
    return weight == NULL ? 1.0f : *weight;
  }
  /*!
   * \return the qid of the instance, this function is always
   *  safe even when qid == NULL
   */
  inline uint64_t get_qid() const {
    return qid == NULL ? 0 : *qid;
  }
  /*!
   * \brief helper function to compute dot product of current
   * \param weight the dense array of weight we want to product
   * \param size the size of the weight vector
   * \tparam V type of the weight vector
   * \return the result of dot product
   */
  template<typename V>
  inline V SDot(const V *weight, size_t size) const {
    V sum = static_cast<V>(0);
    if (value == NULL) {
      for (size_t i = 0; i < length; ++i) {
        CHECK(index[i] < size) << "feature index exceed bound";
        sum += weight[index[i]];
      }
    } else {
      for (size_t i = 0; i < length; ++i) {
        CHECK(index[i] < size) << "feature index exceed bound";
        sum += weight[index[i]] * value[i];
      }
    }
    return sum;
  }
};

/*!
 * \brief a block of data, containing several rows in sparse matrix
 *  This is useful for (streaming-sxtyle) algorithms that scans through rows of data
 *  examples include: SGD, GD, L-BFGS, kmeans
 *
 *  The size of batch is usually large enough so that parallelizing over the rows
 *  can give significant speedup
 * \tparam IndexType type to store the index used in row batch
 * \tparam DType type to store the label and value used in row batch
 */
template<typename IndexType, typename DType = real_t>
struct RowBlock {
  /*! \brief batch size */
  size_t size;
  /*! \brief array[size+1], row pointer to beginning of each rows */
  const size_t *offset;
  /*! \brief array[size] label of each instance */
  const DType *label;
  /*! \brief With weight: array[size] label of each instance, otherwise nullptr */
  const real_t *weight;
  /*! \brief With qid: array[size] session id of each instance, otherwise nullptr */
  const uint64_t *qid;
  /*! \brief field id*/
  const IndexType *field;
  /*! \brief feature index */
  const IndexType *index;
  /*! \brief feature value, can be NULL, indicating all values are 1 */
  const DType *value;
  /*!
   * \brief get specific rows in the batch
   * \param rowid the rowid in that row
   * \return the instance corresponding to the row
   */
  inline Row<IndexType, DType> operator[](size_t rowid) const;
  /*! \return memory cost of the block in bytes */
  inline size_t MemCostBytes(void) const {
    size_t cost = size * (sizeof(size_t) + sizeof(DType));
    if (weight != NULL) cost += size * sizeof(real_t);
    if (qid != NULL) cost += size * sizeof(size_t);
    size_t ndata = offset[size] - offset[0];
    if (field != NULL) cost += ndata * sizeof(IndexType);
    if (index != NULL) cost += ndata * sizeof(IndexType);
    if (value != NULL) cost += ndata * sizeof(DType);
    return cost;
  }
  /*!
   * \brief slice a RowBlock to get rows in [begin, end)
   * \param begin the begin row index
   * \param end the end row index
   * \return the sliced RowBlock
   */
  inline RowBlock Slice(size_t begin, size_t end) const {
    CHECK(begin <= end && end <= size);
    RowBlock ret;
    ret.size = end - begin;
    ret.label = label + begin;
    if (weight != NULL) {
      ret.weight = weight + begin;
    } else {
      ret.weight = NULL;
    }
    if (qid != NULL) {
      ret.qid = qid + begin;
    } else {
      ret.qid = NULL;
    }
    ret.offset = offset + begin;
    ret.field = field;
    ret.index = index;
    ret.value = value;
    return ret;
  }
};

/*!
 * \brief Data structure that holds the data
 * Row block iterator interface that gets RowBlocks
 * Difference between RowBlockIter and Parser:
 *     RowBlockIter caches the data internally that can be used
 *     to iterate the dataset multiple times,
 *     Parser holds very limited internal state and was usually
 *     used to read data only once
 *
 * \sa Parser
 * \tparam IndexType type of index in RowBlock
 * \tparam DType type of label and value in RowBlock
 *  Create function was only implemented for IndexType uint64_t and uint32_t
 *  and DType real_t and int
 */
template<typename IndexType, typename DType = real_t>
class RowBlockIter : public DataIter<RowBlock<IndexType, DType> > {
 public:
  /*!
   * \brief create a new instance of iterator that returns rowbatch
   *  by default, a in-memory based iterator will be returned
   *
   * \param uri the uri of the input, can contain hdfs prefix
   * \param part_index the part id of current input
   * \param num_parts total number of splits
   * \param type type of dataset can be: "libsvm", ...
   *
   * \return the created data iterator
   */
  static RowBlockIter<IndexType, DType> *
  Create(const char *uri,
         unsigned part_index,
         unsigned num_parts,
         const char *type);
  /*! \return maximum feature dimension in the dataset */
  virtual size_t NumCol() const = 0;
};

/*!
 * \brief parser interface that parses input data
 * used to load dmlc data format into your own data format
 * Difference between RowBlockIter and Parser:
 *     RowBlockIter caches the data internally that can be used
 *     to iterate the dataset multiple times,
 *     Parser holds very limited internal state and was usually
 *     used to read data only once
 *
 *
 * \sa RowBlockIter
 * \tparam IndexType type of index in RowBlock
 * \tparam DType type of label and value in RowBlock
 *  Create function was only implemented for IndexType uint64_t and uint32_t
 *  and DType real_t and int
 */
template <typename IndexType, typename DType = real_t>
class Parser : public DataIter<RowBlock<IndexType, DType> > {
 public:
  /*!
  * \brief create a new instance of parser based on the "type"
  *
  * \param uri_ the uri of the input, can contain hdfs prefix
  * \param part_index the part id of current input
  * \param num_parts total number of splits
  * \param type type of dataset can be: "libsvm", "auto", ...
  *
  * When "auto" is passed, the type is decided by format argument string in URI.
  *
  * \return the created parser
  */
  static Parser<IndexType, DType> *
  Create(const char *uri_,
         unsigned part_index,
         unsigned num_parts,
         const char *type);
  /*! \return size of bytes read so far */
  virtual size_t BytesRead(void) const = 0;
  /*! \brief Factory type of the parser*/
  typedef Parser<IndexType, DType>* (*Factory)
      (const std::string& path,
       const std::map<std::string, std::string>& args,
       unsigned part_index,
       unsigned num_parts);
};

/*!
 * \brief registry entry of parser factory
 * \tparam IndexType The type of index
 * \tparam DType The type of label and value
 */
template<typename IndexType, typename DType = real_t>
struct ParserFactoryReg
    : public FunctionRegEntryBase<ParserFactoryReg<IndexType, DType>,
                                  typename Parser<IndexType, DType>::Factory> {};

/*!
 * \brief Register a new distributed parser to dmlc-core.
 *
 * \param IndexType The type of Batch index, can be uint32_t or uint64_t
 * \param DataType The type of Batch label and value, can be real_t or int
 * \param TypeName The typename of of the data.
 * \param FactoryFunction The factory function that creates the parser.
 *
 * \begincode
 *
 *  // define the factory function
 *  template<typename IndexType, typename DType = real_t>
 *  Parser<IndexType, DType>*
 *  CreateLibSVMParser(const char* uri, unsigned part_index, unsigned num_parts) {
 *    return new LibSVMParser(uri, part_index, num_parts);
 *  }
 *
 *  // Register it to DMLC
 *  // Then we can use Parser<uint32_t>::Create(uri, part_index, num_parts, "libsvm");
 *  // to create the parser
 *
 *  DMLC_REGISTER_DATA_PARSER(uint32_t, real_t, libsvm, CreateLibSVMParser<uint32_t>);
 *  DMLC_REGISTER_DATA_PARSER(uint64_t, real_t, libsvm, CreateLibSVMParser<uint64_t>);
 *
 * \endcode
 */
#define DMLC_REGISTER_DATA_PARSER(IndexType, DataType, TypeName, FactoryFunction) \
  DMLC_REGISTRY_REGISTER(ParserFactoryReg<IndexType __DMLC_COMMA DataType>,           \
                         ParserFactoryReg ## _ ## IndexType ## _ ## DataType, TypeName)  \
  .set_body(FactoryFunction)


// implementation of operator[]
template<typename IndexType, typename DType>
inline Row<IndexType, DType>
RowBlock<IndexType, DType>::operator[](size_t rowid) const {
  CHECK(rowid < size);
  Row<IndexType, DType> inst;
  inst.label = label + rowid;
  if (weight != NULL) {
    inst.weight = weight + rowid;
  } else {
    inst.weight = NULL;
  }
  if (qid != NULL) {
    inst.qid = qid + rowid;
  } else {
    inst.qid = NULL;
  }
  inst.length = offset[rowid + 1] - offset[rowid];
  if (field != NULL) {
    inst.field = field + offset[rowid];
  } else {
    inst.field = NULL;
  }
  inst.index = index + offset[rowid];
  if (value == NULL) {
    inst.value = NULL;
  } else {
    inst.value = value + offset[rowid];
  }
  return inst;
}

}  // namespace dmlc
#endif  // DMLC_DATA_H_
