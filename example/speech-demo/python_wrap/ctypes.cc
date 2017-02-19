#include <iostream>

#include "util/table-types.h"
#include "hmm/posterior.h"
#include "nnet/nnet-nnet.h"
#include "cudamatrix/cu-device.h"

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
  typedef RandomAccessPosteriorReader RAPReader;

  namespace nnet1 {
    typedef class Nnet_t_ {
    public:
      Nnet nnet_transf;
      CuMatrix<BaseFloat> feats_transf;
      MatrixF buf;
    } Nnet_t;
  }
}

extern "C" {

  Foo* Foo_new(){ return new Foo(); }
  void Foo_bar(Foo* foo){ foo->bar(); }
  float * Foo_getx(Foo* foo) { return foo->getx(); }
  int Foo_sizex(Foo* foo) { return foo->sizex(); }

  using namespace kaldi;
  using namespace kaldi::nnet1;

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
    return &r->Value(); //despite how dangerous this looks, this is safe because holder maintains object (it's not stack allocated)
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

  /****************************** RAPReader ******************************/

  RAPReader* RAPReader_new_char(char * rspecifier) {
    return new RAPReader(rspecifier);
  }  

  //bool  HasKey (const std::string &key)
  int RAPReader_HasKey(RAPReader* r, char * key) {
    return r->HasKey(key);
  }

  //const T &   Value (const std::string &key)
  int * RAPReader_Value(RAPReader* r, char * key) {
    //return &r->Value(key);
    const Posterior p = r->Value(key);
    int num_rows = p.size();
    if (num_rows == 0) {
      return NULL;
    }

    //std::cout << "num_rows " << num_rows << std::endl;

    int * vals = new int[num_rows];

    for (int row=0; row<num_rows; row++) {
      int num_cols = p.at(row).size();
      if (num_cols != 1) {
        std::cout << "num_cols != 1: " << num_cols << std::endl;
        delete vals;
        return NULL;
      }
      std::pair<int32, BaseFloat> pair = p.at(row).at(0);
      if (pair.second != 1) {
        std::cout << "pair.second != 1: " << pair.second << std::endl;
        delete vals;
        return NULL;
      }
      vals[row] = pair.first;
    }
    
    return vals;
  }

  void RAPReader_DeleteValue(RAPReader* r, int * vals) {
    delete vals;
  }

  //~RandomAccessTableReader ()
  void RAPReader_Delete(RAPReader* r) {
    delete r;
  }

  /****************************** Nnet_t ******************************/

  Nnet_t* Nnet_new(char * filename, float dropout_retention, int crossvalidate) {
    //std::cout << "dropout_retention " << dropout_retention << " crossvalidate " << crossvalidate << std::endl;

    Nnet_t * nnet = new Nnet_t();

    if(strcmp(filename, "") != 0) {
      nnet->nnet_transf.Read(filename);
    }

    if (dropout_retention > 0.0) {
      nnet->nnet_transf.SetDropoutRate(dropout_retention);
    }
    if (crossvalidate) {
      nnet->nnet_transf.SetDropoutRate(1.0);
    }

    return nnet;
  }

  const MatrixF * Nnet_Feedforward(Nnet_t* nnet, MatrixF * inputs) {
    nnet->nnet_transf.Feedforward(CuMatrix<BaseFloat>(*inputs), &nnet->feats_transf);
    nnet->buf.Resize(nnet->feats_transf.NumRows(), nnet->feats_transf.NumCols());
    nnet->feats_transf.CopyToMat(&nnet->buf);
    return &nnet->buf;
  }

  void Nnet_Delete(Nnet_t* nnet) {
    delete nnet;
  }
}
