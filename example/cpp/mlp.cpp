#include <iostream>
using namespace std;

#include "mxnet/ndarray.h"
#include "mxnet/base.h"
#include "mxnet/operator.h"
#include "mxnet/symbolic.h"


#if MSHADOW_USE_CUDA
#define DEV_CTX (mxnet::Context::Create(mxnet::Context::kGPU, 2))
#else
#define DEV_CTX (mxnet::Context::Create(mxnet::Context::kCPU, 0))
#endif

class MLP{
	public:
		mxnet::Symbol LeakyReLULayer(mxnet::Symbol input,std::string name="relu"){

			mxnet::OperatorProperty * leaky_relu_op=mxnet::OperatorProperty::Create("LeakyReLU");

			std::vector<std::pair<std::string,std::string> > relu_config;
			relu_config.push_back(std::make_pair("act_type","leaky"));//rrelu leaky prelu elu
			relu_config.push_back(std::make_pair("slope","0.25"));
			relu_config.push_back(std::make_pair("lower_bound","0.125"));
			relu_config.push_back(std::make_pair("upper_bound","0.334"));
			leaky_relu_op->Init(relu_config);
			std::vector<mxnet::Symbol> sym_vec;

			sym_vec.push_back(input);
			mxnet::Symbol leaky_relu=mxnet::Symbol::Create(leaky_relu_op)(sym_vec,name);
			return leaky_relu;
		}
		mxnet::Symbol FullyConnectedLayer(mxnet::Symbol input,std::string num_hidden="28",std::string name="fc"){

			mxnet::OperatorProperty * fully_connected_op=mxnet::OperatorProperty::Create("FullyConnected");

			std::vector<std::pair<std::string,std::string> > fc_config;
			fc_config.push_back(std::make_pair("num_hidden",num_hidden));
			fc_config.push_back(std::make_pair("no_bias","false"));
			fully_connected_op->Init(fc_config);

			std::vector<mxnet::Symbol> sym_vec;
			sym_vec.push_back(input);
			mxnet::Symbol fc=mxnet::Symbol::Create(fully_connected_op)(sym_vec,name);

			return fc;
		}
		mxnet::Symbol SoftmaxLayer(mxnet::Symbol input,std::string name="softmax"){

			mxnet::OperatorProperty * softmax_output_op=mxnet::OperatorProperty::Create("SoftmaxOutput");

			std::vector<std::pair<std::string,std::string> > config;
			softmax_output_op->Init(config);

			std::vector<mxnet::Symbol> sym_vec;
			sym_vec.push_back(input);
			mxnet::Symbol softmax=mxnet::Symbol::Create(softmax_output_op)(sym_vec,name);

			return softmax;
		}

		void Train(){

			// setup sym network
			mxnet::Symbol sym_x=mxnet::Symbol::CreateVariable("X");
			mxnet::Symbol sym_fc_1=FullyConnectedLayer(sym_x,"128","fc1");
			mxnet::Symbol sym_act_1=LeakyReLULayer(sym_fc_1,"act_1");
			mxnet::Symbol sym_fc_2=FullyConnectedLayer(sym_act_1,"10","fc2");
			mxnet::Symbol sym_act_2=LeakyReLULayer(sym_fc_2,"act_2");
			mxnet::Symbol sym_out=SoftmaxLayer(sym_act_2,"softmax");

			// prepare train data
			mxnet::Context ctx_cpu = mxnet::Context::Create(mxnet::Context::kCPU, 1);
			mxnet::Context ctx_dev = DEV_CTX;// use gpu if possible

			mxnet::NDArray array_x(mshadow::Shape2(100,28),ctx_dev,false);
			mxnet::NDArray array_y(mshadow::Shape1(100),ctx_dev,false);

			mxnet::real_t* aptr_x=new mxnet::real_t[100*28];
			mxnet::real_t* aptr_y=new mxnet::real_t[100];
			for (int i=0;i<100;i++){
				for (int j=0;j<28;j++){
					aptr_x[i*28+j]=i%10*1.0f;
				}
				aptr_y[i]=1;
			}
			array_x.SyncCopyFromCPU(aptr_x,100*28);
			array_x.WaitToRead();
			array_y.SyncCopyFromCPU(aptr_y,100);
			array_y.WaitToRead();
			delete []aptr_x;
			delete []aptr_y;

			// init the parameters
			mxnet::NDArray array_w_1(mshadow::Shape2(128,28),ctx_dev,false);
			mxnet::NDArray array_b_1(mshadow::Shape1(128),ctx_dev,false);
			mxnet::NDArray array_w_2(mshadow::Shape2(10,128),ctx_dev,false);
			mxnet::NDArray array_b_2(mshadow::Shape1(10),ctx_dev,false);

			array_w_1=1.1f;
			array_b_1=0.1f;
			array_w_2=1.1f;
			array_b_2=0.1f;

			// the grads
			mxnet::NDArray array_w_1_g(mshadow::Shape2(128,28),ctx_dev,false);
			mxnet::NDArray array_b_1_g(mshadow::Shape1(128),ctx_dev,false);
			mxnet::NDArray array_w_2_g(mshadow::Shape2(10,128),ctx_dev,false);
			mxnet::NDArray array_b_2_g(mshadow::Shape1(10),ctx_dev,false);


			// Bind
			std::map<std::string,mxnet::Context> g2c;
			//g2c["X"]=ctx_dev;
			//g2c["fc1"]=ctx_dev;
			std::vector<mxnet::NDArray> in_args;
			in_args.push_back(array_x);
			in_args.push_back(array_w_1);
			in_args.push_back(array_b_1);
			in_args.push_back(array_w_2);
			in_args.push_back(array_b_2);
			in_args.push_back(array_y);

			std::vector<mxnet::NDArray> arg_grad_store;
			arg_grad_store.push_back(mxnet::NDArray());
			arg_grad_store.push_back(array_w_1_g);
			arg_grad_store.push_back(array_b_1_g);
			arg_grad_store.push_back(array_w_2_g);
			arg_grad_store.push_back(array_b_2_g);
			arg_grad_store.push_back(mxnet::NDArray());
			std::vector<mxnet::OpReqType> grad_req_type;
			grad_req_type.push_back(mxnet::kNullOp);
			grad_req_type.push_back(mxnet::kWriteTo);
			grad_req_type.push_back(mxnet::kWriteTo);
			grad_req_type.push_back(mxnet::kWriteTo);
			grad_req_type.push_back(mxnet::kWriteTo);
			grad_req_type.push_back(mxnet::kNullOp);
			std::vector<mxnet::NDArray> aux_states;

			cout<<"make the Executor"<<endl;
			mxnet::Executor * exe=mxnet::Executor::Bind(sym_out,
					ctx_dev,
					g2c,
					in_args,
					arg_grad_store,
					grad_req_type,
					aux_states);

			cout<<"Training"<<endl;
			int max_iters=100000;
			mxnet::real_t learning_rate=0.00001;
			for(int iter=0;iter<max_iters;++iter){
				exe->Forward(true);


				if(iter%100==0){
					cout<<"epoch "<<iter<<endl;
					for(int i=0;i<10;++i){
						const std::vector<mxnet::NDArray> &out=exe->outputs();

						mxnet::NDArray c_cpu=out[0].Copy(ctx_cpu);
						c_cpu.WaitToRead();
						mxnet::real_t * cptr = static_cast<mxnet::real_t*>(c_cpu.data().dptr_);
						cout << "prob "<<i<<": "<<cptr[i] << endl;
					}
				}

				//update
				exe->Backward(std::vector<mxnet::NDArray>());
				for(int i=1;i<5;++i){
					in_args[i]-=arg_grad_store[i]*learning_rate;
				}
			}

		}
};

int main(int argc, char ** argv)
{
	MLP mlp;
	mlp.Train();
	return 0;
}
