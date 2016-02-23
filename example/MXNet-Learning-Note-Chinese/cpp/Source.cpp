/*!
*  Copyright (c) 2015 by Contributors
*/
#include <fstream>
#include "cpp_net.hpp"
#include "readmnist.h"
#include "opencv2/opencv.hpp"
/*
* in this example, we use the data from Kaggle mnist match
* get the data from:
* https://www.kaggle.com/c/digit-recognizer
*
*/

using namespace mxnet;
using namespace std;

class MnistCppNet : public mxnet::CppNet {
public:
	void LenetRun() {
		/*
		* LeCun, Yann, Leon Bottou, Yoshua Bengio, and Patrick Haffner.
		* "Gradient-based learning applied to document recognition."
		* Proceedings of the IEEE (1998)
		* */

		/*define the symbolic net*/
		Symbol data = Symbol::CreateVariable("data");
		Symbol conv1 = OperatorSymbol("Convolution", data, "conv1",
			"kernel", mshadow::Shape2(5, 5),
			"num_filter", 20);
		Symbol tanh1 = OperatorSymbol("Activation", conv1, "tanh1",
			"act_type", "tanh");
		Symbol pool1 = OperatorSymbol("Pooling", tanh1, "pool1",
			"pool_type", "max",
			"kernel", mshadow::Shape2(2, 2),
			"stride", mshadow::Shape2(2, 2));
		Symbol conv2 = OperatorSymbol("Convolution", pool1, "conv2",
			"kernel", mshadow::Shape2(5, 5),
			"num_filter", 50);
		Symbol tanh2 = OperatorSymbol("Activation", conv2, "tanh2",
			"act_type", "tanh");
		Symbol pool2 = OperatorSymbol("Pooling", tanh2, "pool2",
			"pool_type", "max",
			"kernel", mshadow::Shape2(2, 2),
			"stride", mshadow::Shape2(2, 2));
		Symbol flatten = OperatorSymbol("Flatten", pool2, "flatten");
		Symbol fc1 = OperatorSymbol("FullyConnected", flatten, "fc1",
			"num_hidden", 500);
		Symbol tanh3 = OperatorSymbol("Activation", fc1, "tanh3",
			"act_type", "tanh");
		Symbol fc2 = OperatorSymbol("FullyConnected", tanh3, "fc2",
			"num_hidden", 10);
		Symbol lenet = OperatorSymbol("SoftmaxOutput", fc2, "softmax");

		/*setup basic configs*/
		int val_fold = 1;
		int W = 28;
		int H = 28;
		int batch_size = 42;
		int max_epoch = 2;
		float learning_rate = 1e-4;

		/*init some of the args*/
		ArgsMap args_map;
		args_map["data"] =
			mxnet::NDArray(mshadow::Shape4(batch_size, 1, W, H), ctx_dev, false);
		/*
		* we can also feed in some of the args other than the input all by
		* ourselves,
		* fc2-weight , fc1-bias for example:
		* */
		args_map["fc1_weight"] =
			mxnet::NDArray(mshadow::Shape2(500, 4 * 4 * 50), ctx_dev, false);
		mxnet::SampleGaussian(0, 1, &args_map["fc1_weight"]);
		args_map["fc2_bias"] = mxnet::NDArray(mshadow::Shape1(10), ctx_dev, false);
		args_map["fc2_bias"] = 0;
		InitArgArrays(lenet, args_map);
		InitOptimizer("ccsgd", "momentum", 0.9, "wd", 1e-4, "rescale_grad", 1.0,
			"clip_gradient", 10);

		/*prepare the data*/
		vector<float> data_vec, label_vec;
		//size_t data_count = Getdata(data_vec, label_vec);
		size_t data_count = GetData(&data_vec, &label_vec);
		const float *dptr = data_vec.data();
		const float *lptr = label_vec.data();
		NDArray data_array = NDArray(mshadow::Shape4(data_count, 1, W, H), ctx_cpu,
			false);  // store in main memory, and copy to
		// device memory while training
		NDArray label_array =
			NDArray(mshadow::Shape1(data_count), ctx_cpu,
			false);  // it's also ok if just store them all in device memory
		data_array.SyncCopyFromCPU(dptr, data_count * W * H);
		label_array.SyncCopyFromCPU(lptr, data_count);
		data_array.WaitToRead();
		label_array.WaitToRead();
	
		std::vector<mxnet::NDArray> argsl;
		Train(data_array, label_array, max_epoch, val_fold, learning_rate,argsl);

		/*mxnet::NDArray out_cpu = argsl[0].Copy(ctx_cpu);
		out_cpu.WaitToRead();
		mxnet::real_t* dptr_out =
			static_cast<mxnet::real_t*>(out_cpu.data().dptr_);
		for (int i = 0; i < 20; i++)
		{
			for (int j = 0; j < 25; j++)
				std::cout << dptr_out[j+25*i];
			std::cout << endl;
		}
		*/
		//output the intermediate layer
		net = tanh1;

		std::vector<mxnet::NDArray> ar, br, au;
		std::vector<mxnet::OpReqType> bw;
		std::vector<mxnet::TShape> ars, ous, aus;
		ArgsMap args_map2;
		args_map2["data"] =
			mxnet::NDArray(mshadow::Shape4(1, 1, W, H), ctx_dev, false);

		vector<string> symv;
		symv = net.ListArguments();
		int ar_number = symv.size();

		std::unordered_map<std::string, mxnet::TShape> known_args;

		ar.assign(argsl.begin(), argsl.begin() + ar_number);

		ar[0] = mxnet::NDArray(mshadow::Shape4(1, 1, W, H), ctx_dev, false);
		known_args["data"] = mshadow::Shape4(1, 1, W, H);

		CHECK(net.InferShape(known_args, &ars, &ous, &aus));

		for (size_t i = 0; i < ars.size(); ++i) {
			br.push_back(mxnet::NDArray(ars[i], ctx_dev, false));
			bw.push_back(mxnet::kWriteTo);
		}
		for (size_t i = 0; i < aus.size(); ++i) {
			au.push_back(mxnet::NDArray(aus[i], ctx_dev, false));
		}
		int k[80];
		for (size_t t= 0; t <80;t++)
		{k[t] = t; }
		for (size_t num = 0; num <80; num++){
			char win_nameI[20], win_name[20];
			ar[0] = data_array.Slice(k[num], k[num] + 1).Copy(ctx_dev);
			//ar[ar.size() - 1] = label_array.Slice(k[num], k[num] + 1).Copy(ctx_dev);
			for (size_t i = 0; i < ar.size(); ++i) {
				ar[i].WaitToRead();
			}

			mxnet::NDArray out_cpuI = ar[0].Copy(ctx_cpu);
			out_cpuI.WaitToRead();
			mxnet::real_t* dptr_outI =
				static_cast<mxnet::real_t*>(out_cpuI.data().dptr_);

			cv::Mat I = cv::Mat::zeros(28, 28, CV_32FC1);;
			float* ptrI = I.ptr<float>(0);
			for (size_t j = 0; j < 28 * 28; j++)
			{
				ptrI[j] = dptr_outI[j];
			}
			cv::normalize(I, I, 0, 255, CV_MINMAX);
			sprintf(win_nameI, "%f imgI%d.jpg", lptr[k[num]], k[num]);
		    cv:imwrite(win_nameI, I);

			std::map<std::string, mxnet::Context> g2c;
			mxnet::Executor* exe1;

			exe1 = mxnet::Executor::Bind(net, ctx_dev, g2c, ar, br,
				bw, au);
			CHECK(exe1);
			exe1->Forward(false);
			const std::vector<mxnet::NDArray>& out = exe1->outputs();
			mshadow::TShape out_shape = out[0].shape();
			mxnet::NDArray out_cpu = out[0].Copy(ctx_cpu);
			out_cpu.WaitToRead();
			mxnet::real_t* dptr_out =
				static_cast<mxnet::real_t*>(out_cpu.data().dptr_);
			/*softmax层输出*/
			/*for (size_t i = 0; i < 10; i++){
				cout << dptr_out[i]<<endl;
			}
			cout << lptr[k[num]]<<endl;
			/*
			/*中间层输出为mat*/
		
			vector<cv::Mat> inter_lay(20);
			for (int i = 0; i < out_shape[1]; i++)
			{
			cv::Mat out_mat = cv::Mat::zeros(out_shape[2], out_shape[3], CV_32FC1);
			float* ptr_mat = out_mat.ptr<float>(0);
			for (size_t j = 0; j < out_shape[2] * out_shape[3]; j++)
			{
				ptr_mat[j] = dptr_out[j + i* out_shape[2] * out_shape[3]];
			}
			cv::normalize(out_mat, out_mat, 0, 255, CV_MINMAX);
			out_mat.copyTo(inter_lay[i]);
		
			//cv::imshow("inter_layer", main_mat);	
			}


			cv::Mat main_mat = cv::Mat::zeros(5 * out_shape[2], 4 * out_shape[3], CV_32FC1);
			cv::Mat main_roi;
			for (size_t i = 0; i <5; i++)
			{
			for (size_t j = 0; j < 4; j++){
			cv::Rect roi(j*out_shape[2], i*out_shape[3], out_shape[2], out_shape[3]);
			main_roi = main_mat(roi);
			
			inter_lay[i * 4 + j].copyTo(main_roi);
			}
			}
			sprintf(win_name, "%f img%d.jpg", lptr[k[num]],k[num]);
			//	cv::imshow("inter_layer", main_mat);

			imwrite(win_name, main_mat);
			
			delete exe1;
			//	cv::waitKey(0);
			/*for (size_t i = 0; i < out_shape[1]; i++)
			{
			sprintf(win_name, "img%d", i );
			cv::imshow(win_name, inter_lay[i]);
			}
			*/
		}
		//cv::Mat plane;

	}

	size_t GetData(vector<float> *data, vector<float> *label) {
		const char *train_data_path = "train.csv";
		ifstream inf(train_data_path);
		if (!inf)
		{
			cout << "open error!" << endl;
			exit(1);
		}
		string line;
		inf >> line;  // ignore the header

		size_t _N = 0;
		while (inf >> line) {
			for (auto &c : line) c = (c == ',') ? ' ' : c;
			stringstream ss;
			ss << line;
			float _data;
			ss >> _data;
			label->push_back(_data);
			while (ss >> _data) data->push_back(_data/256.0);
			_N++;
		}
		inf.close();
		return _N;
	}

	void TrainingCallBack(int iter, const mxnet::Executor *executor) {
		LG << "Iter " << iter << ", accuracy: " << ValAccuracy();
		/*
		* do something every epoch,
		* such as train-data shuffle, train-data augumentation , save the model ,
		* change the learning_rate etc.
		* */
	}

	explicit MnistCppNet(bool use_gpu = false, int dev_id = 0)
		: mxnet::CppNet(use_gpu, dev_id) {}
};

int main(int argc, char const *argv[]) {
	MnistCppNet mnist(true, 0);
	mnist.LenetRun();
	return 0;
}