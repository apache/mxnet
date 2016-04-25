#include <iostream>

#include "matrix/kaldi-matrix.h"
#include "util/table-types.h"

class Foo{
    public:
        Foo() {
            x[0] = 0.5f;
            x[1] = 1.5f;
            x[2] = 2.5f;
            x[3] = 3.5f;
            x[4] = 4.5f;
        }
        void bar(){
            std::cout << "Hello" << std::endl;
        }
        float * getx() {
            return x;
        }
        int sizex() {
            return sizeof(x) / sizeof(float);
        }
    private:
        float x[5];
};

namespace kaldi {
  typedef SequentialBaseFloatMatrixReader SBFMReader;
  typedef Matrix<BaseFloat> MatrixF;
}

extern "C" {

  Foo* Foo_new(){ return new Foo(); }
  void Foo_bar(Foo* foo){ foo->bar(); }
  float * Foo_getx(Foo* foo) { return foo->getx(); }
  int Foo_sizex(Foo* foo) { return foo->sizex(); }
  void Foo_delete(Foo* foo) { delete foo; }

  using namespace kaldi;

  /****************************** SBFMReader ******************************/

  //SequentialTableReader(): impl_(NULL) { }
  SBFMReader* SBFMReader_new() {
    return new SBFMReader();
  }
  //SequentialTableReader(const std::string &rspecifier);
  SBFMReader* SBFMReader_new_char(char * rspecifier) {
    return new SBFMReader(rspecifier);
  }
  //bool Open(const std::string &rspecifier);
  int SBFMReader_Open(SBFMReader* r, char * rspecifier) {
    return r->Open(rspecifier);
  }
  //inline bool Done();
  int SBFMReader_Done(SBFMReader* r) {
    return r->Done();
  }
  //inline std::string Key();
  const char * SBFMReader_Key(SBFMReader* r) {
    return r->Key().c_str();
  }
  //void FreeCurrent();
  void SBFMReader_FreeCurrent(SBFMReader* r) {
    r->FreeCurrent();
  }
  //const T &Value();
  const MatrixF * SBFMReader_Value(SBFMReader* r) {
    return &r->Value();
  }
  //void Next();
  void SBFMReader_Next(SBFMReader* r) {
    r->Next();
  }
  //bool IsOpen() const;
  int SBFMReader_IsOpen(SBFMReader* r) {
    return r->IsOpen();
  }
  //bool Close();
  int SBFMReader_Close(SBFMReader* r) {
    return r->Close();
  }
  //~SequentialTableReader();
  void SBFMReader_Delete(SBFMReader* r) {
    delete r;
  }

  /****************************** MatrixF ******************************/

  //NumRows ()
  int MatrixF_NumRows(MatrixF *m) {
    return m->NumRows();
  }
  //NumCols ()
  int MatrixF_NumCols(MatrixF *m) {
    return m->NumCols();
  }

  //Stride ()
  int MatrixF_Stride(MatrixF *m) {
    return m->Stride();
  }

  void MatrixF_cpy_to_ptr(MatrixF *m, float * dst, int dst_stride) {
    int num_rows = m->NumRows();
    int num_cols = m->NumCols();
    int src_stride = m->Stride();
    int bytes_per_row = num_cols * sizeof(float);

    float * src = m->Data();

    for (int r=0; r<num_rows; r++) {
      memcpy(dst, src, bytes_per_row);
      src += src_stride;
      dst += dst_stride;
    }
  }

  //SizeInBytes ()
  int MatrixF_SizeInBytes(MatrixF *m) {
    return m->SizeInBytes();
  }
  //Data (), Real is usually float32
  const float * MatrixF_Data(MatrixF *m) {
    return m->Data();
  }

} // extern "C"