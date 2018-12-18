/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * This example demonstrates sequence prediction workflow with pre-trained RNN model using MXNet C++ API.
 * The example performs following tasks.
 * 1. Load the pre-trained RNN model,
 * 2. Load the dictionary file that contains word to index mapping.
 * 3. Convert the input string to vector of indices and padded to match the input data length.
 * 4. Run the forward pass and predict the output string.
 * TODO:
 * The purpose of this example is to demonstrate how pre-trained RNN model can be loaded and used
 * to generate an output sequence using C++ API.
 * The example uses a pre-trained RNN model that is trained with the dataset containing speaches
 * given by Obama.
 * The example is intentionally kept specific to a certain model because:
 * 1. Unavailability of pre-trained RNN models that can be successfully imported using C++.
 * 2. C++ API currently does not support bucketing. Hence the shape of input data needs to be
 *    fixed and known while loading the model.
 */

#include <sys/stat.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>
#include "mxnet-cpp/MxNetCpp.h"
#include <opencv2/opencv.hpp>

using namespace mxnet::cpp;



/*
 * class Predictor
 *
 * This class encapsulates the functionality to load the model, process input image and run the forward pass.
 */

class Predictor {
 public:
    Predictor() {}
    Predictor(const std::string& model_json,
              const std::string& model_params,
              const std::string& input_dictionary,
              bool gpu_context_type = false,
              int sequence_length = 35);
    void PredictText(const std::string& input_sequence);
    ~Predictor();

 private:
    void LoadModel(const std::string& model_json_file);
    void LoadParameters(const std::string& model_parameters_file);
    void LoadDictionary(const std::string &input_dictionary);
    inline bool FileExists(const std::string& name) {
        struct stat buffer;
        return (stat(name.c_str(), &buffer) == 0);
    }
    void ConverToIndexVector(const std::string& input,
                      std::vector<float> *input_vector);
    std::map<std::string, NDArray> args_map;
    std::map<std::string, NDArray> aux_map;
    std::map<std::string, int>  wordToInt;
    std::map<int, std::string> intToWord;
    Symbol net;
    Executor *executor;
    Context global_ctx = Context::cpu();
    int sequence_length;
};


/*
 * The constructor takes following parameters as input:
 * 1. model_json:  The RNN model in json formatted file.
 * 2. model_params: File containing model parameters
 * 3. input_dictionary: File containing the word and associated index.
 * 4. sequence_length: Sequence length for which the RNN was trained.
 *
 * The constructor will:
 *  1. Load the model and parameter files.
 *  2. Load the dictionary file to create index to word and word to index maps.
 *  3. Invoke the SimpleBind to bind the input argument to the model and create an executor.
 *
 *  The SimpleBind is expected to be invoked only once.
 */
Predictor::Predictor(const std::string& model_json,
                     const std::string& model_params,
                     const std::string& input_dictionary,
                     bool gpu_context_type,
                     int sequence_length):sequence_length(sequence_length) {
  if (gpu_context_type) {
    global_ctx = Context::gpu();
  }

  /*
   * Load the dictionary file that contains the word and its index.
   * The function creates word to index and index to word map. The maps are used to create index
   * vector for the input sequence as well as converting output prediction to words.
   */
  LoadDictionary(input_dictionary);

  // Load the model
  LoadModel(model_json);

  // Load the model parameters.
  LoadParameters(model_params);

  args_map["data"] = NDArray(Shape(sequence_length, 1), global_ctx, false);
  args_map["label"] = NDArray(Shape(sequence_length, 1), global_ctx, false);

  executor = net.SimpleBind(global_ctx, args_map, std::map<std::string, NDArray>(),
                              std::map<std::string, OpReqType>(), aux_map);
}


/*
 * The following function loads the model from json file.
 */
void Predictor::LoadModel(const std::string& model_json_file) {
  if (!FileExists(model_json_file)) {
    LG << "Model file " << model_json_file << " does not exist";
    throw std::runtime_error("Model file does not exist");
  }
  LG << "Loading the model from " << model_json_file << std::endl;
  net = Symbol::Load(model_json_file);
}


/*
 * The following function loads the model parameters.
 */
void Predictor::LoadParameters(const std::string& model_parameters_file) {
  if (!FileExists(model_parameters_file)) {
    LG << "Parameter file " << model_parameters_file << " does not exist";
    throw std::runtime_error("Model parameters does not exist");
  }
  LG << "Loading the model parameters from " << model_parameters_file << std::endl;
  std::map<std::string, NDArray> paramters;
  NDArray::Load(model_parameters_file, 0, &paramters);
  for (const auto &k : paramters) {
    if (k.first.substr(0, 4) == "aux:") {
      auto name = k.first.substr(4, k.first.size() - 4);
      aux_map[name] = k.second.Copy(global_ctx);
    }
    if (k.first.substr(0, 4) == "arg:") {
      auto name = k.first.substr(4, k.first.size() - 4);
      args_map[name] = k.second.Copy(global_ctx);
    }
  }
  /*WaitAll is need when we copy data between GPU and the main memory*/
  NDArray::WaitAll();
}


/*
 * The following function loads the dictionary file.
 * The function constructs the word to index and index to word maps.
 * These maps will be used to represent words in the input sequence to their indices and
 * conver the indices from predicted output to related words.
 *
 * Ensure to use the same dictionary file that was used for training the network.
 */
void Predictor::LoadDictionary(const std::string& input_dictionary) {
  if (!FileExists(input_dictionary)) {
    LG << "Dictionary file " << input_dictionary << " does not exist";
    throw std::runtime_error("Dictionary file does not exist");
  }
  LG << "Loading the dictionary file.";
  std::ifstream fi(input_dictionary.c_str());
  if (!fi.is_open()) {
    std::cerr << "Error opening dictionary file " << input_dictionary << std::endl;
    assert(false);
  }
  std::string line;
  std::string word;
  int index;
  while (std::getline(fi, line)) {
    std::istringstream stringline(line);
    stringline >> word >> index;
    wordToInt[word] = index;
    intToWord[index] = word;
  }
  fi.close();
}


/*
 * The function populates the input vector with indices from dictionary that are
 * corresponding to the words in the input string.
 */
void Predictor::ConverToIndexVector(const std::string& input, std::vector<float> *input_vector) {
  std::istringstream input_string(input);
  input_vector->clear();
  char delimiter = ' ';
  std::string token;
  int words = 0;
  while (std::getline(input_string, token, delimiter) && (words <= input_vector->size())) {
    input_vector->push_back(static_cast<float>(wordToInt[token]));
    words++;
  }
  return;
}


/*
 * The following function runs the forward pass on the model.
 * The executor is created in the constructor.
 */
void Predictor::PredictText(const std::string& input_text) {
  /*
   * Initialize a vector of length equal to 'sequence_lenght' with 0.
   * Convert the input string to a vector of indices that represent
   * the words in the input string.
   */
  std::vector<float> array(sequence_length, 0);
  ConverToIndexVector(input_text, &array);

  Shape input_shape(sequence_length, 1);
  NDArray input_data = NDArray(input_shape, global_ctx, false);
  input_data.SyncCopyFromCPU(array.data(), input_shape.Size());
  NDArray::WaitAll();

  input_data.CopyTo(&(executor->arg_dict()["data"]));
  NDArray::WaitAll();

  // Run the forward pass.
  executor->Forward(false);

  // The output is available in executor->outputs.
  std::vector<NDArray> outputs = executor->outputs;
  auto arrayout = executor->outputs[4].Copy(global_ctx);
  NDArray::WaitAll();

  /*
   * The output is reshaped to [sequence_length, vocab_size]
   * The vocab_size in this case is equal to the size of dictionary.
   * The output will contain the probability distribution for each of the
   * word in sequence.
   * We will run ArgmaxChannel operator to find out the index with the
   * highest probability. This index will point to word in the predicted
   * output string.
   */
  arrayout = arrayout.Reshape(Shape(sequence_length, -1));
  arrayout = arrayout.ArgmaxChannel();
  NDArray::WaitAll();

  std::ostringstream oss;
  for (std::size_t i = 0; i < sequence_length; ++i) {
    auto charIndex = arrayout.At(0, i);
    oss << intToWord[charIndex] << " ";
  }
  LG << "Output String Predicted by Model: [" << oss.str() << "]";
}


Predictor::~Predictor() {
  if (executor) {
    delete executor;
  }
  MXNotifyShutdown();
}

void printUsage() {
    std::cout << "Usage:" << std::endl;
    std::cout << "simple_rnn " << std::endl
              << "[--input] Input string sequence."  << std::endl
              << "[--gpu]  Specify this option if workflow needs to be run in gpu context "
              << std::endl;
}


void Download_files(const std::vector<std::string> model_files) {
  std::string wget_command = "wget -nc ";
  std::string s3_url = "https://s3.amazonaws.com/mxnet-cpp/RNN_model/";
  for (auto &file : model_files) {
    std::ostringstream oss;
    oss << wget_command << s3_url << file << " -O " << file;
    system(oss.str().c_str());
  }
  return;
}

int main(int argc, char** argv) {
  std::string model_file_json = "./obama-speaks-symbol.json";
  std::string model_file_params ="./obama-speaks-0100.params";
  std::string input_dictionary = "./obama.dictionary.txt";
  std::string input_sequence = "But what they did not understand however was that I had to take "
                "Mr  Keyes seriously for he claimed to speak for my religion and my God";
  int input_sequence_length = 35;
  bool gpu_context_type = false;

  int index = 1;
    while (index < argc) {
      if (strcmp("--input", argv[index]) == 0) {
            index++;
            input_sequence = (index < argc ? argv[index]:input_sequence);
        } else if (strcmp("--gpu", argv[index]) == 0) {
            gpu_context_type = true;
        } else if (strcmp("--help", argv[index]) == 0) {
            printUsage();
            return 0;
        }
      index++;
    }

  /*
   * Download the trained RNN model file, param file and dictionary file.
   * The dictionary file contains word to index mapping.
   * Each line of the dictionary file contains a word and an unique index for that word separated
   * by a space. For example:
   * snippets 11172
   * This dictionary file is created when the RNN model was trained with a particular dataset.
   * Hence the dictionary file is specific to the dataset with which model was trained.
   */
  std::vector<std::string> files;
  files.push_back(model_file_json);
  files.push_back(model_file_params);
  files.push_back(input_dictionary);

  Download_files(files);

  try {
    // Initialize the predictor object
    Predictor predict(model_file_json, model_file_params, input_dictionary, gpu_context_type,
                      input_sequence_length);

    // Run the forward pass to predict the image.
    predict.PredictText(input_sequence);
  } catch (std::runtime_error &error) {
    LG << "Execution failed with ERROR: " << error.what();
  } catch (...) {
    /*
     * If underlying MXNet code has thrown an exception the error message is
     * accessible through MXGetLastError() function.
     */
    LG << "Execution failed with following MXNet error";
    LG << MXGetLastError();
  }

  return 0;
}
