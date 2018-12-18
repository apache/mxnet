/*!
 * Copyright (c) 2017 by Contributors
 * \file main.cpp
 * \brief model-converter main
 * \author HPI-DeepLearning
*/
#include <stdio.h>
#include <libgen.h>
#include <fstream>
#include <dmlc/logging.h>
#include <mxnet/ndarray.h>

#include "../../src/operator/contrib/binary_inference/xnor.h"

#include "rapidjson/document.h"
#include "rapidjson/prettywriter.h"

using mxnet::op::xnor::BITS_PER_BINARY_WORD;
using mxnet::op::xnor::BINARY_WORD;
using namespace std;
using namespace rapidjson;

//======= string constant definition ========//
// layer name related
const string PREFIX_BINARIZED_FILE= "binarized_";
const string POSTFIX_SYM_JSON = "-symbol.json";
const string PREFIX_GRAD_CANCEL = "_contrib_gradcancel";
const string PREFIX_DET_SIGN = "det_sign";
const string PREFIX_Q_CONV = "qconv";
const string PREFIX_Q_DENSE = "qdense";
const string PREFIX_Q_ACTIV = "qactivation";
const string PREFIX_WEIGHT = "weight";
const string PREFIX_BIAS = "bias";
const string PREFIX_FORWARD = "fwd";
// symbol json related
const char* PREFIX_SYM_JSON_NODES = "nodes";
const char* PREFIX_SYM_JSON_NODE_ROW_PTR = "node_row_ptr";
const char* PREFIX_SYM_JSON_ATTRS = "attrs";
const char* PREFIX_SYM_JSON_HEADS = "heads";
const char* PREFIX_SYM_JSON_ARG_NODES = "arg_nodes";

// name of standard convolution and dense layer
const string PREFIX_DENSE = "FullyConnected";
const string PREFIX_CONVOLUTION = "Convolution";
// use this to distinguish arg_nodes : op = 'null'
const string ARG_NODES_OP_PATTERN = "null";

const string PREFIX_BINARY_INFERENCE_CONV_LAYER = "BinaryInferenceConvolution";
const string PREFIX_BINARY_INFERENCE_DENSE_LAYER = "BinaryInferenceFullyConnected";
//==============================================//

/**
 * @brief binarize an NDArray
 * 
 * @param array reference to an NDArray that should be binarized
 */

void convert_to_binary_row(mxnet::NDArray& array) {
  CHECK(array.shape().ndim() >= 2); // second dimension is input depth from prev. layer, needed for next line

  cout << "array shape: " << array.shape() << endl;
  //if(array.shape()[1] < BITS_PER_BINARY_WORD) return;  
  
  CHECK(array.shape()[1] % BITS_PER_BINARY_WORD == 0); // depth from input has to be divisible by 32 (or 64)
  nnvm::TShape binarized_shape(1);
  size_t size = array.shape().Size();
  binarized_shape[0] = size / BITS_PER_BINARY_WORD;
  mxnet::NDArray temp(binarized_shape, mxnet::Context::CPU(), false, mxnet::op::xnor::corresponding_dtype());
  mxnet::op::xnor::get_binary_row((float*) array.data().dptr_, (BINARY_WORD*) temp.data().dptr_, size);
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
  
  cout << "array shape: " << array.shape() << endl;

  //if(array.shape()[0] < BITS_PER_BINARY_WORD) return;
  
  CHECK(array.shape()[0] % BITS_PER_BINARY_WORD == 0); // length of columns has to be divisible by 32 (or 64)
  nnvm::TShape binarized_shape(1);
  size_t size = array.shape().Size();
  binarized_shape[0] = size / BITS_PER_BINARY_WORD;
  mxnet::NDArray temp(binarized_shape, mxnet::Context::CPU(), false, mxnet::op::xnor::corresponding_dtype());
  mxnet::op::xnor::get_binary_col_unrolled((float*) array.data().dptr_, (BINARY_WORD*) temp.data().dptr_, array.shape()[0], array.shape()[1]);
  array = temp;
}

/**
 * @brief filter an array of strings for some keys and then perform task on data
 *
 */

void filter_for(vector<mxnet::NDArray>& data,
                const vector<string>& keys,
                const vector<string>& filter_strings,
                function<void(mxnet::NDArray&)> task) {
  auto containsFilterString = [filter_strings](string line_in_params) {
    auto containsSubString = [line_in_params](string filter_string) {
      return line_in_params.find(filter_string) != string::npos;};
    return find_if(filter_strings.begin(),
                        filter_strings.end(),
                        containsSubString) != filter_strings.end();};

  auto iter = find_if(keys.begin(),
                           keys.end(),
                           containsFilterString);
  int converted = 0;
  //Use a while loop, checking whether iter is at the end of myVector
  //Do a find_if starting at the item after iter, next(iter)
  while (iter != keys.end())
  {
    if ((*iter).find("weight") == string::npos) {
      iter = find_if(next(iter),
                        keys.end(),
                        containsFilterString);
      continue;
    }

    cout << "|- converting weights " << *iter << "..." << endl;

    task(data[iter - keys.begin()]);
    converted++;

    iter = find_if(next(iter),
                        keys.end(),
                        containsFilterString);
  }

  if (converted != filter_strings.size()){
    cout << "Error: The number of found q_conv or q_fc layers doesn't equal the number of converted layers." 
              << endl;
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
int convert_params_file(const string& input_file, const string& output_file, const vector<string> conv_names, const vector<string> fc_names) {
  vector<mxnet::NDArray> data;
  vector<string> keys;

  cout << "loading " << input_file << "..." << endl;
  { // loading params file into data and keys
    unique_ptr<dmlc::Stream> fi(dmlc::Stream::Create(input_file.c_str(), "r"));
    mxnet::NDArray::Load(fi.get(), &data, &keys);
  }

  filter_for(data, keys, conv_names, convert_to_binary_row);
  filter_for(data, keys, fc_names, transpose_and_convert_to_binary_col);

  { // saving params back to *_converted
    unique_ptr<dmlc::Stream> fo(dmlc::Stream::Create(output_file.c_str(), "w"));
    mxnet::NDArray::Save(fo.get(), data, keys);
  }
  cout << "wrote converted params to " << output_file << endl;
  return 0;
}

/**
 * @brief add 'binarized_params_only' attribute to conv and fc layers in mxnet symbol file
 *
 * @param input_file path to mxnet symbol file with QConvolution and QFullyconnected layers
 * @param output_file path to converted symbol file
 * @return success (0) or failure
 */
int convert_json_file(const string& input_fname, const string& output_fname, vector<string>& filters_conv, vector<string>& filters_fc) {
  cout << "loading " << input_fname << "..." << endl;
  string json;
  {
    ifstream stream(input_fname);
    if (!stream.is_open()) {
      cout << "cant find json file at " + input_fname << endl;
      return -1;
    }
    stringstream buffer;
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
  string node_attrs_name = "attrs";
  if (version / 10000 < 1) {
    LOG(INFO) << "detected model saved with mxnet v" << version/10000 << "." << (version/100)%100 << "." << version%100
              << ", using old 'attr' name for layer attributes instead of 'attrs'";
    node_attrs_name = "attr";
  }

  CHECK(d.HasMember(PREFIX_SYM_JSON_NODES));
  rapidjson::Value& nodes = d[PREFIX_SYM_JSON_NODES];
  CHECK(nodes.IsArray());

  for (rapidjson::Value::ValueIterator itr = nodes.Begin(); itr != nodes.End(); ++itr) {
    if (!(itr->HasMember("op") && (*itr)["op"].IsString() &&
            (strcmp((*itr)["op"].GetString(), "QConvolution") == 0 ||
             strcmp((*itr)["op"].GetString(), "QFullyConnected") == 0 ||
             strcmp((*itr)["op"].GetString(), "QConvolution_v1") == 0))) {
      continue;
    }

    CHECK((*itr).HasMember(node_attrs_name.c_str()));
    rapidjson::Value& op_attributes = (*itr)[node_attrs_name.c_str()];
    op_attributes.AddMember("binarized_weights_only", "True", d.GetAllocator());

    CHECK((*itr).HasMember("name"));
    cout << "|- adjusting attributes for " << (*itr)["name"].GetString() << endl;

    if (strcmp((*itr)["op"].GetString(), "QConvolution") == 0 ||
             strcmp((*itr)["op"].GetString(), "QConvolution_v1") == 0) {
      filters_conv.push_back((*itr)["name"].GetString());
    } else if (strcmp((*itr)["op"].GetString(), "QFullyConnected") == 0) {
      filters_fc.push_back((*itr)["name"].GetString());
    }
  }

  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  d.Accept(writer);

  {
    ofstream stream(output_fname);
    if (!stream.is_open()) {
      cout << "cant find json file at " + output_fname << endl;
      return -1;
    }
    string output = buffer.GetString();
    stream << output;
    stream.close();
  }

  cout << "wrote converted json to " << output_fname << endl;

  return 0;
}

/**
 * @brief
    description:
        helper function for printing out "heads", "arg_nodes" and "nodes"
        from the given mxnet symbol json file.
 * @param a json file
 */
void print_rapidjson_doc(string json, string log_prefix="") {
  Document d;  
  d.Parse(json.c_str());

  // print heads
  CHECK(d.HasMember(PREFIX_SYM_JSON_HEADS));
  rapidjson::Value& heads = d[PREFIX_SYM_JSON_HEADS];
  CHECK(heads.IsArray() && heads.Capacity() > 0);
  // logging
  cout << "Info: " << log_prefix.c_str() << "'heads' of input json: " << "[" << "[" 
       << heads[0][0].GetInt() << ", "
       << heads[0][1].GetInt() << ", "
       << heads[0][2].GetInt() 
       << "]" << "]" << endl;

  // print arg_nodes
  CHECK(d.HasMember(PREFIX_SYM_JSON_ARG_NODES));   
  Value& arg_nodes = d[PREFIX_SYM_JSON_ARG_NODES];
  CHECK(arg_nodes.IsArray());
  CHECK(!arg_nodes.Empty());
  // logging
  cout << "Info: " << log_prefix.c_str() << "'arg_nodes' of input json: " << "["; 
  for (int i = 0; i < arg_nodes.Capacity(); ++i)
  {
    cout << arg_nodes[i].GetInt(); 
    if (i < arg_nodes.Capacity()-1)
    {
      cout << ",";
    }
  }
  cout << "]" << endl;

  // print nodes
  CHECK(d.HasMember(PREFIX_SYM_JSON_NODES));
  Value& nodes = d[PREFIX_SYM_JSON_NODES];
  CHECK(nodes.IsArray());
  CHECK(!nodes.Empty());

  cout <<"Info: " << log_prefix.c_str() << "number of nodes:" << nodes.Capacity() << endl;    
  for (int i = 0; i < nodes.Capacity(); ++i)
  {    
    cout <<"Info: " << log_prefix.c_str() << "node index " << i << " : " << nodes[i]["name"].GetString() << endl;
  }
}


/**
 * @brief
    description:
        We modify the json file.
        mxnet symbol json objects to be adapted:
        - nodes: all operators
        - heads: head node
        - arg_nodes: arg nodes, usually 'null' operators.
        - node_row_ptr: not yet found detailed information about this item, 
                        but it seems not affecting the inference
 * @param input_file path to mxnet symbol file with qconv and qdense layers
 * @param output_file path to converted symbol file
 * @return success (0) or failure
 */
int convert_symbol_json(const string& input_fname, const string& output_fname) {
  //logging
  cout << "Info: " <<"Load input 'symbol json' file: "<< input_fname << endl;
  string json;
  {
    ifstream stream(input_fname);
    if (!stream.is_open()) {
      cout << "can't find json file at " + input_fname << endl;
      return -1;
    }
    stringstream buffer;
    buffer << stream.rdbuf();
    json = buffer.str();
  }

  Document d;
  Document::AllocatorType& allocator = d.GetAllocator();
  d.Parse(json.c_str());

  // get heads
  // heads : total num of nodes : [[index last element, 0, 0]]
  CHECK(d.HasMember(PREFIX_SYM_JSON_HEADS));
  rapidjson::Value& heads = d[PREFIX_SYM_JSON_HEADS];
  CHECK(heads.IsArray() && heads.Capacity() > 0);

  // update arg_nodes : contains indices of all "null" op
  CHECK(d.HasMember(PREFIX_SYM_JSON_ARG_NODES));   
  Value& arg_nodes = d[PREFIX_SYM_JSON_ARG_NODES];
  CHECK(arg_nodes.IsArray());
  CHECK(!arg_nodes.Empty());
  
  // check, create nodes
  int retained_op_num = 0;
  CHECK(d.HasMember(PREFIX_SYM_JSON_NODES));
  Value& nodes = d[PREFIX_SYM_JSON_NODES];
  CHECK(nodes.IsArray());
  CHECK(!nodes.Empty());
  Value nodes_new(kArrayType);

  // print the current json docu
  print_rapidjson_doc(json);

  // clear arg_nodes
  arg_nodes.Clear();

  for (Value::ValueIterator itr = nodes.Begin(); itr != nodes.End(); ++itr) {

    CHECK((*itr).HasMember("op"));
    // 1. remove qactivation ops, containing _grad_cancel and det_sign
    if ((string((*itr)["name"].GetString()).find(PREFIX_Q_ACTIV) != string::npos))
      continue;

    // adapt qconv and qdense ops
    CHECK((*itr).HasMember("name"));
    
    bool foundq = false;
    bool retain = false;
    // if qconv or qdense found
    if (string((*itr)["name"].GetString()).find(PREFIX_Q_CONV) != string::npos
        || string((*itr)["name"].GetString()).find(PREFIX_Q_DENSE) != string::npos){
      
      foundq = true;       
      //2.for qconv and qdense, we only retain  'weight', 'bias' and 'fwd'                                    
      if (string((*itr)["name"].GetString()).find(PREFIX_WEIGHT) != string::npos
          || string((*itr)["name"].GetString()).find(PREFIX_BIAS) != string::npos
          || string((*itr)["name"].GetString()).find(PREFIX_FORWARD) != string::npos
         )
        retain = true;  

      // replace convolution and dense operators with binary inference layer
      if ((*itr)["op"].IsString() && 
          string((*itr)["op"].GetString()) == PREFIX_CONVOLUTION){
        (*itr)["op"].SetString(PREFIX_BINARY_INFERENCE_CONV_LAYER.c_str(), allocator);
        //logging
        cout << "Info: " <<"CONVERTING op: '" << (*itr)["name"].GetString() << "' from '"
             << PREFIX_CONVOLUTION.c_str() << "' to '" << PREFIX_BINARY_INFERENCE_CONV_LAYER.c_str() << "'" << endl;
      } 
        
      if ((*itr)["op"].IsString() && 
          string((*itr)["op"].GetString()) == PREFIX_DENSE){
        (*itr)["op"].SetString(PREFIX_BINARY_INFERENCE_DENSE_LAYER.c_str(), allocator);      
        //logging
        cout << "Info: " <<"CONVERTING op: '" << (*itr)["name"].GetString() << "' from '"
             << PREFIX_DENSE.c_str() << "' to '" << PREFIX_BINARY_INFERENCE_DENSE_LAYER.c_str() << "'" << endl;
      }
    }

    if (!foundq || retain){
      // get updated inputs
      CHECK((*itr).HasMember("inputs"));
      CHECK((*itr)["inputs"].IsArray());

      int arr_size = (*itr)["inputs"].Capacity();

      for (int i = 0; i < arr_size; ++i){
        (*itr)["inputs"][i][0].SetInt(retained_op_num - (arr_size - i));        
      }

      // add node      
      nodes_new.PushBack((*itr), allocator);

      // add arg_node
      if ( string((*itr)["op"].GetString()) == ARG_NODES_OP_PATTERN){     
        arg_nodes.PushBack(Value().SetInt(retained_op_num), allocator);        
      }      
      
      //cout << (*itr)["name"].GetString() << endl;     
      retained_op_num++;      
    }
  }  

  // update heads 
  // heads : total num of nodes : [[index last element, 0, 0]]
  heads[0][0].SetInt(retained_op_num - 1);  

  // update nodes  
  nodes = nodes_new;


  // Save output json file
  cout << "Info: " <<"saving new 'symbol json' file to: "<< output_fname << endl;
  rapidjson::StringBuffer buffer;
  rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
  d.Accept(writer);

  {
    ofstream stream(output_fname);
    if (!stream.is_open()) {
      cout << "Error: " << "cant find json file at " + output_fname << endl;
      return -1;
    }
    string output = buffer.GetString();
    stream << output;
    stream.close();

    cout << "Info: " << "converted json file saved!" << endl;
  
    // print the current json docu
    print_rapidjson_doc(output, "updated ");
  }

  return 0;
}

/**
 * @brief convert mxnet param and symbol file to use only binarized weights in conv and fc layers
 *
 */
int main(int argc, char ** argv){
  if (argc < 2 || argc > 3) {
    cout << "usage: " + string(argv[0]) + " <mxnet *.params file>" + " <output (optional)>" << endl;
    cout << "  will binarize the weights of the qconv or qdense layers of your model," << endl;
    cout << "  pack 32(x86 and ARMv7) or 64(x64) values into one and save the result with the prefix 'binarized_'" << endl;
    cout << "<output>: specify the location to store the binarized files. If not specified, the same location as the input model will be used."  << endl;
    return -1;
  }

  const string params_file(argv[1]);
  char *file_copy_basename = strdup(argv[1]); 
  char *file_copy_dirname = strdup(argv[1]);  
  const string path(dirname(file_copy_dirname));
  const string params_file_name(basename(file_copy_basename));
  string out_path;
  if(argc == 3)
    out_path = argv[2];
  if(out_path.empty())
    out_path = path;
  free(file_copy_basename);
  free(file_copy_dirname);

  string base_name = params_file_name;
  base_name.erase(base_name.rfind('-')); // watchout if no '-'

  const string json_file_name(path + "/"                + base_name + "-symbol.json");
  const string param_out_fname(out_path + "/" + "binarized_" + params_file_name);
  const string json_out_fname(out_path + "/" + "binarized_" + base_name + "-symbol.json");

  //logging
  cout << "Info: " <<"Load input '.params' file: "<< params_file << endl;
  
  if (int ret = convert_symbol_json(json_file_name, json_out_fname) != 0) {
    return ret;
  }

  if (int ret = convert_params_file(params_file, param_out_fname, filters_conv, filters_fc) != 0) {
    return ret;
  }
 
  cout << "Info: " <<"Output '.params' file path: "<< param_out_fname << endl;
  return 0;
}
