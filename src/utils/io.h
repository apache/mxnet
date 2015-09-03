#ifndef CXXNET_UTILS_IO_H_
#define CXXNET_UTILS_IO_H_
/*!
 * \file io.h
 * \brief definition of abstract stream interface for IO
 * \author Bing Xu Tianqi Chen
 */
#include "./utils.h"
#include <dmlc/io.h>
#include <string>
#include <algorithm>
#include <cstring>

namespace cxxnet {
namespace utils {
typedef dmlc::Stream IStream;
typedef dmlc::SeekStream ISeekStream;

/*! \brief a in memory buffer that can be read and write as stream interface */
struct MemoryBufferStream : public ISeekStream {
 public:
  MemoryBufferStream(std::string *p_buffer)
      : p_buffer_(p_buffer) {
    curr_ptr_ = 0;
  }
  virtual ~MemoryBufferStream(void) {}
  virtual size_t Read(void *ptr, size_t size) {
    CHECK(curr_ptr_ <= p_buffer_->length())
          << " read can not have position excceed buffer length";
    size_t nread = std::min(p_buffer_->length() - curr_ptr_, size);
    if (nread != 0) memcpy(ptr, &(*p_buffer_)[0] + curr_ptr_, nread);
    curr_ptr_ += nread;
    return nread;
  }
  virtual void Write(const void *ptr, size_t size) {
    if (size == 0) return;
    if (curr_ptr_ + size > p_buffer_->length()) {
      p_buffer_->resize(curr_ptr_+size);
    }
    memcpy(&(*p_buffer_)[0] + curr_ptr_, ptr, size);
    curr_ptr_ += size;
  }
  virtual void Seek(size_t pos) {
    curr_ptr_ = static_cast<size_t>(pos);
  }
  virtual size_t Tell(void) {
    return curr_ptr_;
  }

 private:
  /*! \brief in memory buffer */
  std::string *p_buffer_;
  /*! \brief current pointer */
  size_t curr_ptr_;
}; // class MemoryBufferStream

/*! \brief implementation of file i/o stream */
class StdFile: public ISeekStream {
 public:
  /*! \brief constructor */
  StdFile(const char *fname, const char *mode) {
    Open(fname, mode);
  }
  StdFile() {}
  virtual ~StdFile(void) {
    this->Close();
  }
  virtual void Open(const char *fname, const char *mode) {
    fp_ = utils::FopenCheck(fname, mode);
    fseek(fp_, 0L, SEEK_END);
    sz_ = ftell(fp_);
    fseek(fp_, 0L, SEEK_SET);
  }
  virtual size_t Read(void *ptr, size_t size) {
    return fread(ptr, size, 1, fp_);
  }
  virtual void Write(const void *ptr, size_t size) {
    fwrite(ptr, size, 1, fp_);
  }
  virtual void Seek(size_t pos) {
    fseek(fp_, pos, SEEK_SET);
  }
  virtual size_t Tell(void) {
    return static_cast<size_t>(ftell(fp_));
  }
  inline void Close(void) {
    if (fp_ != NULL){
      fclose(fp_); fp_ = NULL;
    }
  }
  inline size_t Size() {
    return sz_;
  }
 private:
  FILE *fp_;
  size_t sz_;
}; // class StdFile

/*! \brief Basic page class */
class BinaryPage {
 public:
  /*! \brief page size 64 MB */
  static const size_t kPageSize = 64 << 18;
 public:
  /*! \brief memory data object */
  struct Obj{
    /*! \brief pointer to the data*/
    void  *dptr;
    /*! \brief size */
    size_t sz;
    Obj(void * dptr, size_t sz) : dptr(dptr), sz(sz){}
  };
 public:
  /*! \brief constructor of page */
  BinaryPage(void)  {
    data_ = new int[kPageSize];
    utils::Check(data_ != NULL, "fail to allocate page, out of space");
    this->Clear();
  };
  ~BinaryPage() {
    if (data_) delete [] data_;
  }
  /*!
   * \brief load one page form instream
   * \return true if loading is successful
   */
  inline bool Load(utils::IStream &fi) {
    return fi.Read(&data_[0], sizeof(int)*kPageSize) !=0;
  }
  /*! \brief save one page into outstream */
  inline void Save(utils::IStream &fo) {
    fo.Write(&data_[0], sizeof(int)*kPageSize);
  }
  /*! \return number of elements */
  inline int Size(void){
    return data_[0];
  }
  /*! \brief Push one binary object into page
   *  \param fname file name of obj need to be pushed into
   *  \return false or true to push into
   */
  inline bool Push(const Obj &dat) {
    if(this->FreeBytes() < dat.sz + sizeof(int)) return false;
    data_[ Size() + 2 ] = data_[ Size() + 1 ] + dat.sz;
    memcpy(this->offset(data_[ Size() + 2 ]), dat.dptr, dat.sz);
    ++ data_[0];
    return true;
  }
  /*! \brief Clear the page */
  inline void Clear(void) {
    memset(&data_[0], 0, sizeof(int) * kPageSize);
  }
  /*!
   * \brief Get one binary object from page
   *  \param r r th obj in the page
   */
  inline Obj operator[](int r) {
    CHECK(r < Size());
    return Obj(this->offset(data_[ r + 2 ]),  data_[ r + 2 ] - data_[ r + 1 ]);
  }
 private:
  /*! \return number of elements */
  inline size_t FreeBytes(void) {
    return (kPageSize - (Size() + 2)) * sizeof(int) - data_[ Size() + 1 ];
  }
  inline void* offset(int pos) {
    return (char*)(&data_[0]) + (kPageSize*sizeof(int) - pos);
  }
 private:
  //int data_[ kPageSize ];
  int *data_;
};  // class BinaryPage
}  // namespace utils
}  // namespace cxxnet
#endif
