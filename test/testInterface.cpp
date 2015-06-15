#include <iostream>
using namespace std;

#include "engine/dagengine.h"
#include "common/all_ops.h"

DAGEngine * engine = new SingleThreadDAGEngine;
StorageManager * storage = new NaiveStorageManager;

/*! \brief a wierd network, mixing layers with NArrays */
int main()
{
  const static int M = 1024;
  const static int N = 512;
  const static int K = 1000;
  FloatT delta = 0.24;
  FloatT * customA = new FloatT[M * N];
  DummyLayer layer(delta);
  NArray layerIn({M, N}, customA);
  NArray layerOut({ M, N });
  FF({ layerIn }, { &layerOut }, &layer);
  NArray x = layerIn + layerOut;
  DummyNArrayLayer l2;
  NArray y({M, N});
  FF({ layerIn }, { &y }, &l2);

  NArray a({ 1024, 768 });
  NArray b({ 1024, 768 });
  NArray c({ 1024, 768 });
  NArray d = a + b + c;

  WaitForAll();

  return 0;
}

