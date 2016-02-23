/*!
*  Copyright (c) 2015 by Contributors
* \file cpp_net.hpp
* \brief A simple interface for cpp.
*/
#ifndef CPP_NET_HPP_YP6CSQNU
#define CPP_NET_HPP_YP6CSQNU

#include <mxnet/ndarray.h>
#include <mxnet/base.h>
#include <mxnet/operator.h>
#include <mxnet/symbolic.h>
#include <mxnet/optimizer.h>

#include <sstream>
#include <utility>
#include <map>
#include <string>
#include <vector>

namespace mxnet {
	/*! \brief The cpp interface */
	class  CppNet {
	public:
		explicit CppNet(bool use_gpu = false, int dev_id = 0) {
			ctx_cpu = mxnet::Context::Create(mxnet::Context::kCPU, dev_id);
			if (use_gpu)
				ctx_dev = mxnet::Context::Create(mxnet::Context::kGPU, dev_id);
			else
				ctx_dev = ctx_cpu;

			exe = nullptr;
			optimizer = nullptr;
		}

		template <typename T, typename... Args>
		void InitOptimizer(const char* optimizer_name, const char* config_key,
			const T& config_value, const Args&... args) {
			optimizer = mxnet::Optimizer::Create(optimizer_name);

			std::vector<std::pair<std::string, std::string> > config_dict;
			_GetConfigDict(&config_dict, config_key, config_value, args...);
			optimizer->Init(config_dict);
		}

		mxnet::Symbol OperatorSymbol(const char* op_name, mxnet::Symbol input,
			const char* symbol_name) {
			mxnet::OperatorProperty* sym_op_prop =
				mxnet::OperatorProperty::Create(op_name);

			std::vector<std::pair<std::string, std::string> > config_dict;
			sym_op_prop->Init(config_dict);

			std::vector<mxnet::Symbol> sym_vec;
			sym_vec.push_back(input);

			mxnet::Symbol sym_op =
				mxnet::Symbol::Create(sym_op_prop)(sym_vec, symbol_name);
			return sym_op;
		}
		template <typename T, typename... Args>
		mxnet::Symbol OperatorSymbol(const char* op_name, mxnet::Symbol input,
			const char* symbol_name, const char* config_key,
			const T& config_value, const Args&... args) {
			mxnet::OperatorProperty* sym_op_prop =
				mxnet::OperatorProperty::Create(op_name);

			std::vector<std::pair<std::string, std::string> > config_dict;
			_GetConfigDict(&config_dict, config_key, config_value, args...);
			sym_op_prop->Init(config_dict);

			std::vector<mxnet::Symbol> sym_vec;
			sym_vec.push_back(input);

			mxnet::Symbol sym_op =
				mxnet::Symbol::Create(sym_op_prop)(sym_vec, symbol_name);
			return sym_op;
		}

		virtual void TrainingCallBack(int iter, const mxnet::Executor* executor) {
			/*overide this*/
		}

		float ValAccuracy() {
			int start_index = 0;
			int val_num = val_data.shape()[0];

			int right_count = 0;
			int all_count = 0;
			int batch_size = in_args[0].shape()[0];

			in_args[0] =
				val_data.Slice(start_index, start_index + batch_size).Copy(ctx_dev);
			in_args[in_args.size() - 1] =
				val_label.Slice(start_index, start_index + batch_size).Copy(ctx_dev);
			in_args[0].WaitToRead();
			in_args[in_args.size() - 1].WaitToRead();
			while (start_index < val_num) {
				delete exe;
				exe = mxnet::Executor::Bind(net, ctx_dev, g2c, in_args, arg_grad_store,
					grad_req_type, aux_states);

				CHECK(exe);

				exe->Forward(false);

				start_index += batch_size;
				if (start_index < val_num) {
					if (start_index + batch_size >= val_num)
						start_index = val_num - batch_size;
					in_args[0] =
						val_data.Slice(start_index, start_index + batch_size).Copy(ctx_dev);
					in_args[in_args.size() - 1] =
						val_label.Slice(start_index, start_index + batch_size)
						.Copy(ctx_dev);
				}

				const std::vector<mxnet::NDArray>& out = exe->outputs();
				mxnet::NDArray out_cpu = out[0].Copy(ctx_cpu);
				mxnet::NDArray label_cpu =
					val_label.Slice(start_index - batch_size, start_index).Copy(ctx_cpu);

				in_args[0].WaitToRead();
				in_args[in_args.size() - 1].WaitToRead();
				out_cpu.WaitToRead();
				label_cpu.WaitToRead();

				mxnet::real_t* dptr_out =
					static_cast<mxnet::real_t*>(out_cpu.data().dptr_);
				mxnet::real_t* dptr_label =
					static_cast<mxnet::real_t*>(label_cpu.data().dptr_);
				for (int i = 0; i < batch_size; ++i) {
					float label = dptr_label[i];
					int cat_num = out_cpu.shape()[1];
					float p_label = 0, max_p = dptr_out[i * cat_num];
					for (int j = 0; j < cat_num; ++j) {
						float p = dptr_out[i * cat_num + j];
						if (max_p < p) {
							p_label = j;
							max_p = p;
						}
					}
					if (label == p_label) right_count++;
				}
				all_count += batch_size;
			}
			return right_count * 1.0 / all_count;
		}

		void Train(mxnet::NDArray data_array, mxnet::NDArray label_array,
			int max_epoches, int val_fold, float start_learning_rate, std::vector<mxnet::NDArray> &argsl) {
			/*prepare ndarray*/
			learning_rate = start_learning_rate;
			size_t data_count = data_array.shape()[0];
			size_t val_data_count = data_count * val_fold / 10;
			size_t train_data_count = data_count - val_data_count;
			train_data = data_array.Slice(0, train_data_count);
			train_label = label_array.Slice(0, train_data_count);
			val_data = data_array.Slice(train_data_count, data_count);
			val_label = label_array.Slice(train_data_count, data_count);
			size_t batch_size = in_args[0].shape()[0];

			/*start the training*/
			for (int iter = 0; iter < max_epoches; ++iter) {
				CHECK(optimizer);
				size_t start_index = 0;
				in_args[0] =
					train_data.Slice(start_index, start_index + batch_size).Copy(ctx_dev);
				in_args[in_args.size() - 1] =
					train_label.Slice(start_index, start_index + batch_size)
					.Copy(ctx_dev);
				in_args[0].WaitToRead();
				in_args[in_args.size() - 1].WaitToRead();
				while (start_index < train_data_count) {
					/*rebind the excutor*/
					delete exe;
					exe = mxnet::Executor::Bind(net, ctx_dev, g2c, in_args, arg_grad_store,
						grad_req_type, aux_states);

					CHECK(exe);

					exe->Forward(true);
					exe->Backward(std::vector<mxnet::NDArray>());

					start_index += batch_size;
					if (start_index < train_data_count) {
						if (start_index + batch_size >= train_data_count)
							start_index = train_data_count - batch_size;
						in_args[0] = train_data.Slice(start_index, start_index + batch_size)
							.Copy(ctx_dev);
						in_args[in_args.size() - 1] =
							train_label.Slice(start_index, start_index + batch_size)
							.Copy(ctx_dev);
					}

					for (size_t i = 1; i < in_args.size() - 1; ++i) {
						optimizer->Update(i, &in_args[i], &arg_grad_store[i], learning_rate);
					}
					for (size_t i = 1; i < in_args.size() - 1; ++i) {
						in_args[i].WaitToRead();
					}
					in_args[0].WaitToRead();
					in_args[in_args.size() - 1].WaitToRead();
				}

				/*call every iter*/
				TrainingCallBack(iter, exe);
			}
		
			argsl.assign(in_args.begin(), in_args.end());
		
		}

		typedef std::map<std::string, mxnet::NDArray> ArgsMap;
		void InitArgArrays(mxnet::Symbol net_sym, ArgsMap args_map) {
			net = net_sym;

			const auto& arg_name_list = net.ListArguments();
			std::unordered_map<std::string, mxnet::TShape> known_arg_shapes;

			std::vector<mxnet::TShape> arg_shapes;
			std::vector<mxnet::TShape> out_shapes;
			std::vector<mxnet::TShape> aux_shapes;

			for (const auto& arg_name : arg_name_list) {
				if (args_map.find(arg_name) != args_map.end()) {
					in_args.push_back(args_map[arg_name]);
					known_arg_shapes[arg_name] = args_map[arg_name].shape();

				}
				else {
					in_args.push_back(mxnet::NDArray());
				}
			}

			CHECK(net.InferShape(known_arg_shapes, &arg_shapes, &out_shapes,
				&aux_shapes));

			for (size_t i = 0; i < in_args.size(); ++i) {
				if (in_args[i].shape().ndim() == 0) {
					in_args[i] = mxnet::NDArray(arg_shapes[i], ctx_dev, false);
					// TODO(zhangchen-qinyinghua) a better initialize
					mxnet::SampleUniform(0, 0.01, &in_args[i]);
				}
				arg_grad_store.push_back(mxnet::NDArray(arg_shapes[i], ctx_dev, false));
				grad_req_type.push_back(mxnet::kWriteTo);
			}
			for (size_t i = 0; i < aux_shapes.size(); ++i) {
				aux_states.push_back(mxnet::NDArray(aux_shapes[i], ctx_dev, false));
			}

			for (size_t i = 0; i < in_args.size(); ++i) {
				in_args[i].WaitToRead();
			}

		}

	protected:
		mxnet::Context ctx_cpu;
		mxnet::Context ctx_dev;

		template <typename T>
		void _GetConfigDict(
			std::vector<std::pair<std::string, std::string> >* config_dict,
			const char* key = "", const T& value = "") {
			std::string value_s;
			std::stringstream ss;
			ss << value;
			ss >> value_s;
			config_dict->push_back(std::make_pair(key, value_s));
		}
		template <typename T, typename... Args>
		void _GetConfigDict(
			std::vector<std::pair<std::string, std::string> >* config_dict,
			const char* key, const T& value, Args... args) {
			std::string value_s;
			std::stringstream ss;
			ss << value;
			ss >> value_s;
			config_dict->push_back(std::make_pair(key, value_s));

			_GetConfigDict(config_dict, args...);
		}

		/*! \brief the symbolic net*/
		mxnet::Symbol net;
		mxnet::Executor* exe;
		mxnet::Optimizer* optimizer;
		float learning_rate;
		std::map<std::string, mxnet::Context> g2c;
		/*! \brief the input args feed to the net*/
		std::vector<mxnet::NDArray> in_args;
		/*! \brief the array store the grad*/
		std::vector<mxnet::NDArray> arg_grad_store;
		std::vector<mxnet::OpReqType> grad_req_type;
		std::vector<mxnet::NDArray> aux_states;

		mxnet::NDArray train_data;
		mxnet::NDArray train_label;
		mxnet::NDArray val_data;
		mxnet::NDArray val_label;
	};  // class CppNet
}  // namespace mxnet

#endif /* end of include guard: CPP_NET_HPP_YP6CSQNU */