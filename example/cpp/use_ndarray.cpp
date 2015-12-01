#include <iostream>
using namespace std;

#include "mxnet/ndarray.h"
#include "mxnet/base.h"

int main(int argc, char ** argv)
{
  // using NDArray interface
  const int m = 3;
  const int n = 5;
  mshadow::TShape f;
  f = mshadow::Shape2(m, n);
  mxnet::Context d = mxnet::Context::Create(mxnet::Context::kCPU, 1);
  mxnet::NDArray a(f, d, false);
  mxnet::NDArray b(f, d, false);
  mxnet::real_t* aptr = static_cast<mxnet::real_t*>(a.data().dptr_);
  mxnet::real_t* bptr = static_cast<mxnet::real_t*>(b.data().dptr_);
  for (int i = 0; i < m; i++)
  {
    for (int j = 0; j < n; j++)
    {
      aptr[i*n + j] = i*n + j;
      bptr[i*n + j] = i*n + j;
    }
  }
  mxnet::NDArray c = a + b;
  // this is important, wait for the execution to complete before reading
  c.WaitToRead();
  mxnet::real_t* cptr = static_cast<mxnet::real_t*>(c.data().dptr_);
  cout << cptr[m * n - 1] << endl;
  return 0;
}
