/*!
 * Copyright (c) 2017 by Contributors
 * \file main.cpp
 * \brief model-converter main
 * \author HPI-DeepLearning
*/
#include <stdio.h>
#include <libgen.h>
#include <fstream>

#include <mxnet/ndarray.h>

#include "../src/xnor_cpu.h"

#include "rapidjson/document.h"
#include "rapidjson/writer.h"

using mxnet::op::xnor_cpu::BITS_PER_BINARY_WORD;
using mxnet::op::xnor_cpu::BINARY_WORD;

/**
 * @brief binarize an NDArray
 * 
 * @param array reference to an NDArray that should be binarized
 */

void convert_to_binary_row(mxnet::NDArray& array) {
  CHECK(array.shape().ndim() >= 2); // second dimension is input depth from prev. layer, needed for next line

  std::cout << "array shape: " << array.shape() << std::endl;
  //if(array.shape()[1] < BITS_PER_BINARY_WORD) return;  
  
  CHECK(array.shape()[1] % BITS_PER_BINARY_WORD == 0); // depth from input has to be divisible by 32 (or 64)
  nnvm::TShape binarized_shape(1);
  size_t size = array.shape().Size();
  binarized_shape[0] = size / BITS_PER_BINARY_WORD;
  mxnet::NDArray temp(binarized_shape, mxnet::Context::CPU(), false, mxnet::op::xnor_cpu::corresponding_dtype());
  mxnet::op::xnor_cpu::get_binary_row((float*) array.data().dptr_, (BINARY_WORD*) temp.data().dptr_, size);
  array = temp;
}

/**
 * @brief transposes an NDArray
 *
 * @param array reference to an NDArray that should be transposed
 */

void transpose(mxnet::NDArray& array) {
  CHECK(array.shape().ndim() == 2);
  nnvm::TShape tansposed_shape(2);
  int rows = array.shape()[0];
  int cols = array.shape()[1];
  tansposed_shape[0] = cols;
  tansposed_shape[1] = rows;
  mxnet::NDArray temp(tansposed_shape, mxnet::Context::CPU(), false, array.dtype());
  MSHADOW_REAL_TYPE_SWITCH(array.dtype(), DType, {
    for (int row = 0; row < rows; row++) {
      for (int col = 0; col < cols; col++) {
        ((DType*)temp.data().dptr_)[col * rows + row] = ((DType*)array.data().dptr_)[row * cols + col];
      }
    }
  })
  array = temp;
}

/**
 * @brief transpose and then binarize an array column wise
 *
 * @param array reference to an NDArray that should be binarized
 */

void transpose_and_convert_to_binary_col(mxnet::NDArray& array) {
  transpose(array);
  CHECK(array.shape().ndim() == 2); // since we binarize column wise, we need to know no of rows and columns
  
  std::cout << "array shape: " << array.shape() << std::endl;

  //if(array.shape()[0] < BITS_PER_BINARY_WORD) return;
  
  CHECK(array.shape()[0] % BITS_PER_BINARY_WORD == 0); // length of columns has to be divisible by 32 (or 64)
  nnvm::TShape binarized_shape(1);
  size_t size = array.shape().Size();
  binarized_shape[0] = size / BITS_PER_BINARY_WORD;
  mxnet::NDArray temp(binarized_shape, mxnet::Context::CPU(), false, mxnet::op::xnor_cpu::corresponding_dtype());
  mxnet::op::xnor_cpu::get_binary_col_unrolled((float*) array.data().dptr_, (BINARY_WORD*) temp.data().dptr_, array.shape()[0], array.shape()[1]);
  array = temp;
}

/**
 * @brief filter an array of strings for some keys and then perform task on data
 *
 */

void filter_for(std::vector<mxnet::NDArray>& data,
                const std::vector<std::string>& keys,
                const std::vector<std::string>& filter_strings,
                std::function<void(mxnet::NDArray&)> task) {
  auto containsFilterString = [filter_strings](std::string line_in_params) {
    auto containsSubString = [line_in_params](std::string filter_string) {
      return line_in_params.find(filter_string) != std::string::npos;};
    return std::find_if(filter_strings.begin(),
                        filter_strings.end(),
                        containsSubString) != filter_strings.end();};

  auto iter = std::find_if(keys.begin(),
                           keys.end(),
                           containsFilterString);
  int converted = 0;
  //Use a while loop, checking whether iter is at the end of myVector
  //Do a find_if starting at the item after iter, std::next(iter)
  while (iter != keys.end())
  {
    if ((*iter).find("weight") == std::string::npos) {
      iter = std::find_if(std::next(iter),
                        keys.end(),
                        containsFilterString);
      continue;
    }

    std::cout << "|- converting weights " << *iter << "..." << std::endl;

    task(data[iter - keys.begin()]);
    converted++;

    iter = std::find_if(std::next(iter),
                        keys.end(),
                        containsFilterString);
  }

  if (converted != filter_strings.size()){
    std::cout << "Error: The number of found q_conv or q_fc layers doesn't equal the number of converted layers." 
              << std::endl;
  }
}

/**
 * @brief convert convolutional and fully connected layers of mxnet params file to binary format
 *
 * @param input_file path to mxnet params file with QConvolution and QFullyconnected layers
 * @param output_file path to converted file
 * @param filter_strings list of strings with arrays to convert
 * @return success (0) or failure
 */
int convert_params_file(const std::string& input_file, const std::string& output_file, const std::vector<std::string> conv_names, const std::vector<std::string> fc_names) {
  std::vector<mxnet::NDArray> data;
  std::vector<std::string> keys;

  std::cout << "loading " << input_file << "..." << std::endl;
  { // loading params file into data and keys
    std::unique_ptr<dmlc::Stream> fi(dmlc::Stream::Create(input_file.c_str(), "r"));
    mxnet::NDArray::Load(fi.get(), &data, &keys);
  }

  filter_for(data, keys, conv_names, convert_to_binary_row);
  filter_for(data, keys, fc_names, transpose_and_convert_to_binary_col);

  { // saving params back to *_converted
    std::unique_ptr<dmlc::Stream> fo(dmlc::Stream::Create(output_file.c_str(), "w"));
    mxnet::NDArray::Save(fo.get(), data, keys);
  }
  std::cout << "wrote converted params to " << output_file << std::endl;
  return 0;
}

/**
 * @brief add 'binarized_params_only' attribute to conv and fc layers in mxnet symbol file
 *
 * @param input_file path to mxnet symbol file with QConvolution and QFullyconnected layers
 * @param output_file path to converted symbol file
 * @return success (0) or failure
 */
int convert_json_file(const std::string& input_fname, const std::string& output_fname, std::vector<std::string>& filters_conv, std::vector<std::string>& filters_fc) {
  std::cout << "loading " << input_fname << "..." << std::endl;
  std::string json;
  {
    std::ifstream stream(input_fname);
    if (!stream.is_open()) {
      std::cout << "cant find json file at " + input_fname << std::endl;
      return -1;
    }
    std::stringstream buffer;
    buffer << stream.rdbuf();
    json = buffer.str();
  }

  rapidjson::Document d;
  d.Parse(json.c_str());

  // detecting mxnet version, starting with v1.0.0 (?) they switched from key 'attr' to 'attrs'
  CHECK(d.HasMember("attrs"));
  rapidjson::Value::ConstMemberIterator itr = d["attrs"].FindMember("mxnet_version");
  CHECK(itr != d["attrs"].MemberEnd());
  CHECK(itr->value.IsArray());
  CHECK(itr->value.Size() == 2);
  int version = itr->value[1].GetInt();
  std::string node_attrs_name = "attrs";
  if (version / 10000 < 1) {
    LOG(INFO) << "detected model saved with mxnet v" << version/10000 << "." << (version/100)%100 << "." << version%100
              << ", using old 'attr' name for layer attributes instead of 'attrs'";
    node_attrs_name = "attr";
  }

  CHECK(d.HasMember("nodes"));
  rapidjson::Value& nodes = d["nodes"];
  CHECK(nodes.IsArray());

  for (rapidjson::Value::ValueIterator itr = nodes.Begin(); itr != nodes.End(); ++itr) {
    if (!(itr->HasMember("op") && (*itr)["op"].IsString() &&
            (std::strcmp((*itr)["op"].GetString(), "QConvolution") == 0 ||
             std::strcmp((*itr)["op"].GetString(), "QFullyConnected") == 0 ||
             std::strcmp((*itr)["op"].GetString(), "QConvolution_v1") == 0))) {
      continue;
    }

    CHECK((*itr).HasMember(node_attrs_name.c_str()));
    rapidjson::Value& op_attributes = (*itr)[node_attrs_name.c_str()];
    op_attributes.AddMember("binarized_weights_only", "True", d.GetAllocator());

    CHECK((*itr).HasMember("name"));
    std::cout << "|- adjusting attributes for " << (*itr)["name"].GetString() << std::endl;

    if (std::strcmp((*itr)["op"].GetString(), "QConvolution") == 0 ||
             std::strcmp((*itr)["op"].GetString(), "QConvolution_v1") == 0) {
      filters_conv.push_back((*itr)["name"].GetString());
    } else if (std::strcmp((*itr)["op"].GetString(), "QFullyConnected") == 0) {
      filters_fc.push_back((*itr)["name"].GetString());
    }
  }

  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  d.Accept(writer);

  {
    std::ofstream stream(output_fname);
    if (!stream.is_open()) {
      std::cout << "cant find json file at " + output_fname << std::endl;
      return -1;
    }
    std::string output = buffer.GetString();
    stream << output;
    stream.close();
  }

  std::cout << "wrote converted json to " << output_fname << std::endl;

  return 0;
}

/**
 * @brief convert mxnet param and symbol file to use only binarized weights in conv and fc layers
 *
 */
int main(int argc, char ** argv){
  if (argc != 2) {
    std::cout << "usage: " + std::string(argv[0]) + " <mxnet *.params file>" << std::endl;
    std::cout << "  will binarize the weights of the Convolutional Layers of your model," << std::endl;
    std::cout << "  pack 32(x86 and ARMv7) or 64(x64) values into one and save the result with the prefix 'binarized_'" << std::endl;
    return -1;
  }

  const std::string params_file(argv[1]);
  char *file_copy_basename = strdup(argv[1]); 
  char *file_copy_dirname = strdup(argv[1]);
  const std::string path(dirname(file_copy_dirname));
  const std::string params_file_name(basename(file_copy_basename));
  free(file_copy_basename);
  free(file_copy_dirname);

  std::string base_name = params_file_name;
  base_name.erase(base_name.rfind('-')); // watchout if no '-'
  const std::string output_name(path + "/" + "binarized_" + params_file_name);

  const std::string json_file_name(path + "/"                + base_name + "-symbol.json");
  const std::string json_out_fname(path + "/" + "binarized_" + base_name + "-symbol.json");

  std::vector<std::string> filters_conv;
  std::vector<std::string> filters_fc;

  if (int ret = convert_json_file(json_file_name, json_out_fname, filters_conv, filters_fc) != 0) {
    return ret;
  }

  if (int ret = convert_params_file(params_file, output_name, filters_conv, filters_fc) != 0) {
    return ret;
  }

  return 0;
}
