#include "mshadow/tensor.h"
#include "old/tensor.h"
#include "assert.h"
#include <cstring>

using mshadow::index_t;
template<typename T>
void Print(T const & ist) {
  for (int i = 0; i < ist.size(0); ++i) {
    for (int j = 0; j < ist.size(1); ++j) {
      printf("%.2f ", ist[i][j]);
    }
    printf("\n");
  }
}

bool Check(mshadow::TensorContainer<mshadow::cpu, 2, float> &mct, \
           Xmshadow::TensorContainer<Xmshadow::cpu, 2> &xct) {
  for (index_t i = 0; i < mct.size(0); ++i) {
    for (index_t j = 0; j < mct.size(1); ++j) {
      assert(mct[i][j] == xct[i][j]);
    }
  }
  return true;
}

template<typename xpua, typename xpub>
void RunTask() {
  const int ksize = 3;
  const int kstride = 2;
  const int X = 6;
  Xmshadow::TensorContainer<Xmshadow::cpu, 4> xsrc(Xmshadow::Shape4(1, 1, X, X));
  mshadow::TensorContainer<mshadow::cpu, 4> src(mshadow::Shape4(1, 1, X, X));

  for (int i = 0; i < X; ++i) {
    for (int j = 0; j < X; ++j) {
      xsrc[0][0][i][j] = i * 0.1f + j * 0.2f;
      src[0][0][i][j] = i * 0.1f + j * 0.2f;
    }
  }
  Xmshadow::TensorContainer<xpub, 4> xin(Xmshadow::Shape4(1, 1, X, X));
  mshadow::TensorContainer<xpua, 4> in(mshadow::Shape4(1, 1, X, X));

  mshadow::Copy(in, src);
  Xmshadow::Copy(xin, xsrc);

  Xmshadow::TensorContainer<xpub, 2> xtmp_col;
  mshadow::TensorContainer<xpua, 2> tmp_col;
  

  index_t oheight  = (in.size(2) - ksize)/kstride + 1;
  index_t owidth   = (in.size(3) - ksize)/kstride + 1;
  index_t nbatch   = in.size(0);

  
  xtmp_col.Resize( Xmshadow::Shape2( xin.shape[2]*ksize*ksize, nbatch*oheight*owidth ) );
  tmp_col.Resize(mshadow::Shape2(in.size(1)*ksize*ksize, nbatch*oheight*owidth));
  xtmp_col = Xmshadow::expr::unpack_patch2col( xin, ksize, kstride );
  tmp_col = mshadow::expr::unpack_patch2col(in, ksize, ksize, kstride);

  Xmshadow::TensorContainer<Xmshadow::cpu, 2> xtc;
  mshadow::TensorContainer<mshadow::cpu, 2> tc;

  xtc.Resize( Xmshadow::Shape2( xin.shape[2]*ksize*ksize, nbatch*oheight*owidth ) );
  tc.Resize(mshadow::Shape2(in.size(1)*ksize*ksize, nbatch*oheight*owidth));

  mshadow::Copy(tc, tmp_col);
  Xmshadow::Copy(xtc, xtmp_col);
  if (Check(tc, xtc)) {
    printf("Pass\n");
  }
  
}

int main(int argc, char** argv) {
  if (argc < 2) {
    printf("Usage: dev\n");
    exit(-1);
  }
  if (!strcmp(argv[1], "cpu")) {
    RunTask<mshadow::cpu, Xmshadow::cpu>();
  } else {
    RunTask<mshadow::gpu, Xmshadow::gpu>();
  }
}