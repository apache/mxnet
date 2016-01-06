/*!
 *  Copyright (c) 2015 by Contributors
 * \file c_predict_api.cc
 * \brief C predict API of mxnet
 */
#include <dmlc/base.h>
#include <dmlc/memory_io.h>
#include <mxnet/c_predict_api.h>
#include <mxnet/symbolic.h>
#include <mxnet/ndarray.h>
#include <memory>
#include <unordered_set>
#include <unordered_map>
#include "./c_api_error.h"

using namespace mxnet;

// predictor interface
struct MXAPIPredictor {
  // output arrays
  std::vector<NDArray> out_arrays;
  // argument arrays
  std::vector<NDArray> arg_arrays;
  // output shapes
  std::vector<TShape> out_shapes;
  // key to arguments
  std::unordered_map<std::string, size_t> key2arg;
  // executor
  std::unique_ptr<Executor> exec;
};

struct MXAPINDList {
  std::vector<std::string> keys;
  std::vector<TShape> shapes;
  std::vector<size_t> indptr;
  std::vector<mx_float> data;
};

int MXPredCreate(const char* symbol_json_str,
                 const void* param_bytes,
                 int param_size,
                 int dev_type, int dev_id,
                 mx_uint num_input_nodes,
                 const char** input_keys,
                 const mx_uint* input_shape_indptr,
                 const mx_uint* input_shape_data,
                PredictorHandle* out) {
  return MXPredCreatePartialOut(
      symbol_json_str,
      param_bytes,
      param_size,
      dev_type,
      dev_id,
      num_input_nodes,
      input_keys,
      input_shape_indptr,
      input_shape_data,
      0,
      NULL,
      out);
}

int MXPredCreatePartialOut(const char* symbol_json_str,
                           const void* param_bytes,
                           int param_size,
                           int dev_type, int dev_id,
                           mx_uint num_input_nodes,
                           const char** input_keys,
                           const mx_uint* input_shape_indptr,
                           const mx_uint* input_shape_data,
                           mx_uint num_output_nodes,
                           const char** output_keys,
                           PredictorHandle* out) {
  MXAPIPredictor* ret = new MXAPIPredictor();
  API_BEGIN();
  Symbol sym;
  // load in the symbol.
  {
    std::string json = symbol_json_str;
    std::istringstream is(json);
    dmlc::JSONReader reader(&is);
    sym.Load(&reader);
  }
  // looks likely to output the internal results
  if (num_output_nodes != 0) {
    Symbol internal = sym.GetInternals();
    std::vector<std::string> all_out = internal.ListOutputs();
    std::vector<Symbol> out_syms(num_output_nodes);
    for (mx_uint i = 0; i < num_output_nodes; ++i) {
      std::string out_key(output_keys[i]);
      out_key += "_output";
      for (size_t j = 0; j < all_out.size(); ++j) {
        if (all_out[j] == out_key) {
          out_syms[i] = internal[j];
          break;
        }
        CHECK_NE(j, all_out.size() - 1) << "didn't find node name: " << out_key;
      }
    }
    sym = Symbol::CreateGroup(out_syms);
  }

  // load the parameters
  std::unordered_map<std::string, NDArray> arg_params, aux_params;
  {
    std::unordered_set<std::string> arg_names, aux_names;
    std::vector<std::string> arg_names_vec = sym.ListArguments();
    std::vector<std::string> aux_names_vec = sym.ListAuxiliaryStates();
    for (size_t i = 0; i < arg_names_vec.size(); ++i) {
      arg_names.insert(arg_names_vec[i]);
    }
    for (size_t i = 0; i < aux_names_vec.size(); ++i) {
      aux_names.insert(aux_names_vec[i]);
    }
    std::vector<NDArray> data;
    std::vector<std::string> names;
    dmlc::MemoryFixedSizeStream fi((void*)param_bytes, param_size);  // NOLINT(*)
    NDArray::Load(&fi, &data, &names);
    CHECK_EQ(names.size(), data.size())
        << "Invalid param file format";
    for (size_t i = 0; i < names.size(); ++i) {
      if (!strncmp(names[i].c_str(), "aux:", 4)) {
        std::string name(names[i].c_str() + 4);
        if (aux_names.count(name) != 0) {
          aux_params[name] = data[i];
        }
      }
      if (!strncmp(names[i].c_str(), "arg:", 4)) {
        std::string name(names[i].c_str() + 4);
        if (arg_names.count(name) != 0) {
          arg_params[name] = data[i];
        }
      }
    }
  }

  // shape inference and bind
  std::unordered_map<std::string, TShape> known_shape;
  for (mx_uint i = 0; i < num_input_nodes; ++i) {
    known_shape[std::string(input_keys[i])] =
        TShape(input_shape_data + input_shape_indptr[i],
               input_shape_data + input_shape_indptr[i + 1]);
  }
  std::vector<TShape> arg_shapes;
  std::vector<std::string> arg_names = sym.ListArguments();
  std::vector<std::string> aux_names = sym.ListAuxiliaryStates();
  std::vector<TShape> out_shapes(sym.ListOutputs().size());
  std::vector<TShape> aux_shapes(aux_names.size());
  for (size_t i = 0; i < arg_names.size(); ++i) {
    std::string key = arg_names[i];
    ret->key2arg[key] = i;
    if (known_shape.count(key) != 0) {
      arg_shapes.push_back(known_shape[key]);
    } else {
      arg_shapes.push_back(TShape());
    }
  }
  CHECK(sym.InferShape(&arg_shapes, &out_shapes, &aux_shapes))
      << "The shape information of is not enough to get the shapes";
  ret->out_shapes = out_shapes;
  Context ctx = Context::Create(static_cast<Context::DeviceType>(dev_type), dev_id);

  std::vector<NDArray> arg_arrays, aux_arrays;
  for (size_t i = 0; i < arg_shapes.size(); ++i) {
    NDArray nd = NDArray(arg_shapes[i], ctx);
    if (arg_params.count(arg_names[i]) != 0) {
      CopyFromTo(arg_params[arg_names[i]], &nd);
    }
    arg_arrays.push_back(nd);
  }
  for (size_t i = 0; i < aux_shapes.size(); ++i) {
    NDArray nd = NDArray(aux_shapes[i], ctx);
    if (aux_params.count(aux_names[i]) != 0) {
      CopyFromTo(aux_params[aux_names[i]], &nd);
    }
    aux_arrays.push_back(nd);
  }
  ret->arg_arrays = arg_arrays;
  // bind
  {
    std::map<std::string, Context> ctx_map;
    std::vector<NDArray> grad_store(arg_arrays.size());
    std::vector<OpReqType> grad_req(arg_arrays.size(), kNullOp);
    ret->exec.reset(Executor::Bind(sym, ctx, ctx_map,
                                   arg_arrays,
                                   grad_store, grad_req,
                                   aux_arrays));
    ret->out_arrays = ret->exec->outputs();
  }
  *out = ret;
  API_END_HANDLE_ERROR(delete ret);
}

int MXPredGetOutputShape(PredictorHandle handle,
                         mx_uint out_index,
                         mx_uint** shape_data,
                         mx_uint* shape_ndim) {
  MXAPIPredictor* p = static_cast<MXAPIPredictor*>(handle);
  API_BEGIN();
  CHECK_LT(out_index, p->out_arrays.size())
      << "Index exceed number of outputs";
  *shape_data = p->out_shapes[out_index].data();
  *shape_ndim = p->out_shapes[out_index].ndim();
  API_END();
}

int MXPredSetInput(PredictorHandle handle,
                   const char* key,
                   const mx_float* data,
                   mx_uint size) {
  MXAPIPredictor* p = static_cast<MXAPIPredictor*>(handle);
  API_BEGIN();
  auto it = p->key2arg.find(key);
  if (it == p->key2arg.end()) {
    LOG(FATAL) << "cannot find input key " << key;
  }
  NDArray& nd = p->arg_arrays[it->second];
  nd.SyncCopyFromCPU(data, size);
  API_END();
}

int MXPredForward(PredictorHandle handle) {
  MXAPIPredictor* p = static_cast<MXAPIPredictor*>(handle);
  API_BEGIN();
  p->exec->Forward(false);
  API_END();
}

int MXPredPartialForward(PredictorHandle handle, int step, int* step_left) {
  MXAPIPredictor* p = static_cast<MXAPIPredictor*>(handle);
  API_BEGIN();
  p->exec->PartialForward(false, step, step_left);
  API_END();
}

int MXPredGetOutput(PredictorHandle handle,
                    mx_uint index,
                    mx_float* data,
                    mx_uint size) {
  MXAPIPredictor* p = static_cast<MXAPIPredictor*>(handle);
  API_BEGIN();
  CHECK_LT(index, p->out_arrays.size())
      << "Output index out of range";
  const NDArray& nd = p->out_arrays[index];
  nd.SyncCopyToCPU(data, size);
  API_END();
}

int MXPredFree(PredictorHandle handle) {
  API_BEGIN();
  delete static_cast<MXAPIPredictor*>(handle);
  API_END();
}

int MXNDListCreate(const char* nd_file_bytes,
                   int nd_file_size,
                   NDListHandle *out,
                   mx_uint* out_length) {
  MXAPINDList* ret = new MXAPINDList();
  API_BEGIN();
  std::vector<NDArray> arrays;
  dmlc::MemoryFixedSizeStream fi((void*)nd_file_bytes, nd_file_size);  // NOLINT(*)
  NDArray::Load(&fi,
                &(arrays),
                &(ret->keys));
  if (ret->keys.size() == 0) {
    ret->keys.resize(arrays.size());
  }
  ret->indptr.push_back(0);
  for (size_t i = 0; i < arrays.size(); ++i) {
    TShape shape = arrays[i].shape();
    size_t begin = ret->data.size();
    size_t size = shape.Size();
    ret->shapes.push_back(shape);
    ret->data.resize(begin + size);
    arrays[i].SyncCopyToCPU(dmlc::BeginPtr(ret->data) + begin, size);
    ret->indptr.push_back(begin + size);
  }
  *out = ret;
  *out_length = static_cast<mx_uint>(arrays.size());
  API_END();
}

int MXNDListGet(NDListHandle handle,
                mx_uint index,
                const char** out_key,
                const mx_float** out_data,
                const mx_uint** out_shape,
                mx_uint* out_ndim) {
  MXAPINDList* p = static_cast<MXAPINDList*>(handle);
  API_BEGIN();
  CHECK_LT(index, p->shapes.size())
      << "Index out of range";
  *out_key = p->keys[index].c_str();
  *out_data = dmlc::BeginPtr(p->data) + p->indptr[index];
  *out_shape = p->shapes[index].data();
  *out_ndim = p->shapes[index].ndim();
  API_END();
}

int MXNDListFree(NDListHandle handle) {
  API_BEGIN();
  delete static_cast<MXAPINDList*>(handle);
  API_END();
}
