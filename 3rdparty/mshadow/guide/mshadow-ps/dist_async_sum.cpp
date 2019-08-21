#include "./dist_async_sum-inl.h"

int CreateServerNode(int argc, char *argv[]) {
  mshadow::ps::MShadowServerNode<float> server(argc, argv);
  return 0;
}


int WorkerNodeMain(int argc, char *argv[]) {
  return Run<mshadow::cpu>(argc, argv);
}
