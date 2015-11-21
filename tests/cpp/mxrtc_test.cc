#include <cstdio>
#include <gtest/gtest.h>
#include <dmlc/logging.h>

#if MXNET_USE_CUDA

#include <mxnet/mxrtc.h>

using namespace mxnet;
using namespace std;

TEST(MXRtc, Basic_GPU) {
	vector<pair<string, NDArray*> > input;
	vector<pair<string, NDArray*> > output;
	mshadow::TShape shape = mshadow::Shape1(100);
	input.push_back(pair<string, NDArray*>("x", new NDArray(shape, Context::GPU(0))));
	output.push_back(pair<string, NDArray*>("z", new NDArray(shape, Context::GPU(0))));
	*input[0].second = 10;
	*output[0].second = 1;
	float * buff = new float[100];

	MXRtc mod("test", input, output, "z[threadIdx.x] = x[threadIdx.x]*2.0;\n");


	vector<NDArray*> nd_input;
	nd_input.push_back(input[0].second);
	vector<NDArray*> nd_output;
	nd_output.push_back(output[0].second);
	mod.push(nd_input, nd_output, 100, 1, 1, 1, 1, 1);
	nd_output[0]->WaitToRead();
	nd_output[0]->SyncCopyToCPU(buff, 100);
	for (int i = 0; i < 100; ++i) {
		CHECK_EQ(buff[0], 20.0);
	}

	input.clear();
	output.clear();
	input.push_back(pair<string, NDArray*>("x", new NDArray(shape, Context::GPU(1))));
	output.push_back(pair<string, NDArray*>("z", new NDArray(shape, Context::GPU(1))));
	*input[0].second = 10;
	*output[0].second = 1;

	nd_input.clear();
	nd_input.push_back(input[0].second);
	nd_output.clear();
	nd_output.push_back(output[0].second);
	mod.push(nd_input, nd_output, 100, 1, 1, 1, 1, 1);
	nd_output[0]->WaitToRead();
	nd_output[0]->SyncCopyToCPU(buff, 100);
	for (int i = 0; i < 100; ++i) {
		CHECK_EQ(buff[0], 20.0);
	}
}
#else
TEST(MXRtc, Basic_GPU) {
	LOG(INFO) << "CUDA disabled. Test ignored";
}	
#endif  // MXNET_USE_CUDA