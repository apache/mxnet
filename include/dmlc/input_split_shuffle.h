/*!
 *  Copyright (c) 2016 by Contributors
 * \file input_split_shuffle.h
 * \brief base class to construct input split with global shuffling
 * \author Yifeng Geng
 */
#ifndef DMLC_INPUT_SPLIT_SHUFFLE_H_
#define DMLC_INPUT_SPLIT_SHUFFLE_H_

#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include <memory>

namespace dmlc {
/*! \brief class to construct input split with global shuffling */
class InputSplitShuffle : public InputSplit {
 public:
  // destructor
  virtual ~InputSplitShuffle(void) { source_.reset(); }
  // implement BeforeFirst
  virtual void BeforeFirst(void) {
    if (num_shuffle_parts_ > 1) {
      std::shuffle(shuffle_indexes_.begin(), shuffle_indexes_.end(), trnd_);
      int idx = shuffle_indexes_[0] + part_index_ * num_shuffle_parts_;
      source_->ResetPartition(idx, num_parts_ * num_shuffle_parts_);
      cur_shuffle_idx_ = 0;
    } else {
      source_->BeforeFirst();
    }
  }
  virtual void HintChunkSize(size_t chunk_size) {
    source_->HintChunkSize(chunk_size);
  }
  virtual size_t GetTotalSize(void) {
    return source_->GetTotalSize();
  }
  // implement next record
  virtual bool NextRecord(Blob *out_rec) {
    if (num_shuffle_parts_ > 1) {
      if (!source_->NextRecord(out_rec)) {
        if (cur_shuffle_idx_ == num_shuffle_parts_ - 1) {
          return false;
        }
        ++cur_shuffle_idx_;
        int idx =
            shuffle_indexes_[cur_shuffle_idx_] + part_index_ * num_shuffle_parts_;
        source_->ResetPartition(idx, num_parts_ * num_shuffle_parts_);
        return NextRecord(out_rec);
      } else {
        return true;
      }
    } else {
      return source_->NextRecord(out_rec);
    }
  }
  // implement next chunk
  virtual bool NextChunk(Blob* out_chunk) {
    if (num_shuffle_parts_ > 1) {
      if (!source_->NextChunk(out_chunk)) {
        if (cur_shuffle_idx_ == num_shuffle_parts_ - 1) {
          return false;
        }
        ++cur_shuffle_idx_;
        int idx =
            shuffle_indexes_[cur_shuffle_idx_] + part_index_ * num_shuffle_parts_;
        source_->ResetPartition(idx, num_parts_ * num_shuffle_parts_);
        return NextChunk(out_chunk);
      } else {
        return true;
      }
    } else {
      return source_->NextChunk(out_chunk);
    }
  }
  // implement ResetPartition.
  virtual void ResetPartition(unsigned rank, unsigned nsplit) {
    CHECK(nsplit == num_parts_) << "num_parts is not consistent!";
    int idx = shuffle_indexes_[0] + rank * num_shuffle_parts_;
    source_->ResetPartition(idx, nsplit * num_shuffle_parts_);
    cur_shuffle_idx_ = 0;
  }
  /*!
   * \brief constructor
   * \param uri the uri of the input, can contain hdfs prefix
   * \param part_index the part id of current input
   * \param num_parts total number of splits
   * \param type type of record
   *   List of possible types: "text", "recordio"
   *     - "text":
   *         text file, each line is treated as a record
   *         input split will split on '\\n' or '\\r'
   *     - "recordio":
   *         binary recordio file, see recordio.h
   * \param num_shuffle_parts number of shuffle chunks for each split
   * \param shuffle_seed shuffle seed for chunk shuffling
   */
  InputSplitShuffle(const char* uri,
                    unsigned part_index,
                    unsigned num_parts,
                    const char* type,
                    unsigned num_shuffle_parts,
                    int shuffle_seed)
      : part_index_(part_index),
        num_parts_(num_parts),
        num_shuffle_parts_(num_shuffle_parts),
        cur_shuffle_idx_(0) {
    for (unsigned i = 0; i < num_shuffle_parts_; i++) {
      shuffle_indexes_.push_back(i);
    }
    trnd_.seed(kRandMagic_ + part_index_ + num_parts_ + num_shuffle_parts_ +
               shuffle_seed);
    std::shuffle(shuffle_indexes_.begin(), shuffle_indexes_.end(), trnd_);
    int idx = shuffle_indexes_[cur_shuffle_idx_] + part_index_ * num_shuffle_parts_;
    source_.reset(
        InputSplit::Create(uri, idx , num_parts_ * num_shuffle_parts_, type));
  }
  /*!
   * \brief factory function:
   *  create input split with chunk shuffling given a uri
   * \param uri the uri of the input, can contain hdfs prefix
   * \param part_index the part id of current input
   * \param num_parts total number of splits
   * \param type type of record
   *   List of possible types: "text", "recordio"
   *     - "text":
   *         text file, each line is treated as a record
   *         input split will split on '\\n' or '\\r'
   *     - "recordio":
   *         binary recordio file, see recordio.h
   * \param num_shuffle_parts number of shuffle chunks for each split
   * \param shuffle_seed shuffle seed for chunk shuffling
   * \return a new input split
   * \sa InputSplit::Type
   */
  static InputSplit* Create(const char* uri,
                            unsigned part_index,
                            unsigned num_parts,
                            const char* type,
                            unsigned num_shuffle_parts,
                            int shuffle_seed) {
    CHECK(num_shuffle_parts > 0) << "number of shuffle parts should be greater than zero!";
    return new InputSplitShuffle(
        uri, part_index, num_parts, type, num_shuffle_parts, shuffle_seed);
  }

 private:
  // magic nyumber for seed
  static const int kRandMagic_ = 666;
  /*! \brief random engine */
  std::mt19937 trnd_;
  /*! \brief inner inputsplit */
  std::unique_ptr<InputSplit> source_;
  /*! \brief part index */
  unsigned part_index_;
  /*! \brief number of parts */
  unsigned num_parts_;
  /*! \brief the number of block for shuffling*/
  unsigned num_shuffle_parts_;
  /*! \brief current shuffle block index */
  unsigned cur_shuffle_idx_;
  /*! \brief shuffled indexes */
  std::vector<int> shuffle_indexes_;
};
}  // namespace dmlc
#endif  // DMLC_INPUT_SPLIT_SHUFFLE_H_
