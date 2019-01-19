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
 * This example demonstrates sentiment prediction workflow with pre-trained RNN model using MXNet C++ API.
 * The example performs following tasks.
 * 1. Load the pre-trained RNN model,
 * 2. Load the dictionary file that contains word to index mapping.
 * 3. Convert the input string to vector of indices and padded to match the input data length.
 * 4. Run the forward pass and predict the output string.
 * The example uses a pre-trained RNN model that is trained with the IMDB dataset.
 */

#include <sys/stat.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>
#include <sstream>
#include "mxnet-cpp/MxNetCpp.h"

using namespace mxnet::cpp;

static const int DEFAULT_NUM_WORDS = 5;
static const char DEFAULT_S3_URL[] = "https://s3.amazonaws.com/mxnet-cpp/RNN_model/";

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
              bool use_gpu = false,
              int num_words = DEFAULT_NUM_WORDS);
    float PredictSentiment(const std::string &input_sequence);
    ~Predictor();

 private:
    void LoadModel(const std::string& model_json_file);
    void LoadParameters(const std::string& model_parameters_file);
    void LoadDictionary(const std::string &input_dictionary);
    inline bool FileExists(const std::string& name) {
        struct stat buffer;
        return (stat(name.c_str(), &buffer) == 0);
    }
    int ConverToIndexVector(const std::string& input,
                      std::vector<float> *input_vector);
    int GetIndexForOutputSymbolName(const std::string& output_symbol_name);
    float GetIndexForWord(const std::string& word);
    std::map<std::string, NDArray> args_map;
    std::map<std::string, NDArray> aux_map;
    std::map<std::string, int>  wordToIndex;
    Symbol net;
    Executor *executor;
    Context global_ctx = Context::cpu();
    int num_words;
};


/*
 * The constructor takes the following parameters as input:
 * 1. model_json:  The RNN model in json formatted file.
 * 2. model_params: File containing model parameters
 * 3. input_dictionary: File containing the word and associated index.
 * 4. num_words: Number of words which will be used to predict the sentiment.
 *
 * The constructor:
 *  1. Loads the model and parameter files.
 *  2. Loads the dictionary file to create index to word and word to index maps.
 *  3. Invokes the SimpleBind to bind the input argument to the model and create an executor.
 *
 *  The SimpleBind is expected to be invoked only once.
 */
Predictor::Predictor(const std::string& model_json,
                     const std::string& model_params,
                     const std::string& input_dictionary,
                     bool use_gpu,
                     int num_words):num_words(num_words) {
  if (use_gpu) {
    global_ctx = Context::gpu();
  }

  /*
   * Load the dictionary file that contains the word and its index.
   * The function creates word to index and index to word map. The maps are used to create index
   * vector for the input sentence.
   */
  LoadDictionary(input_dictionary);

  // Load the model
  LoadModel(model_json);

  // Load the model parameters.
  LoadParameters(model_params);

  args_map["data0"] = NDArray(Shape(num_words, 1), global_ctx, false);
  args_map["data1"] = NDArray(Shape(1), global_ctx, false);

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
  std::map<std::string, NDArray> parameters;
  NDArray::Load(model_parameters_file, 0, &parameters);
  for (const auto &k : parameters) {
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
 * These maps will be used to represent words in the input sentence to their indices.
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
    wordToIndex[word] = index;
  }
  fi.close();
}


/*
 * The function returns the index associated with the word in the dictionary.
 * If the word is not present, the index representing "<unk>" is returned.
 * If the "<unk>" is not present then 0 is returned.
 */
float Predictor::GetIndexForWord(const std::string& word) {
  if (wordToIndex.find(word) == wordToIndex.end()) {
    if (wordToIndex.find("<unk>") == wordToIndex.end())
      return 0;
    else
      return static_cast<float>(wordToIndex["<unk>"]);
  }
  return static_cast<float>(wordToIndex[word]);
}

/*
 * The function populates the input vector with indices from the dictionary that
 * correspond to the words in the input string.
 */
int Predictor::ConverToIndexVector(const std::string& input, std::vector<float> *input_vector) {
  std::istringstream input_string(input);
  input_vector->clear();
  const char delimiter = ' ';
  std::string token;
  size_t words = 0;
  while (std::getline(input_string, token, delimiter) && (words <= input_vector->size())) {
    LG << token << " " << static_cast<float>(wordToIndex[token]);
    input_vector->push_back(GetIndexForWord(token));
    words++;
  }
  return words;
}


/*
 * The function returns the index at which the given symbol name will appear
 * in the output vector of NDArrays obtained after running the forward pass on the executor.
 */
int Predictor::GetIndexForOutputSymbolName(const std::string& output_symbol_name) {
  int index = 0;
  for (const std::string op : net.ListOutputs()) {
    if (op == output_symbol_name) {
      return index;
    } else {
      index++;
    }
  }
  throw std::runtime_error("The output symbol name can not be found");
}


/*
 * The following function runs the forward pass on the model.
 * The executor is created in the constructor.
 */
float Predictor::PredictSentiment(const std::string& input_text) {
  /*
   * Initialize a vector of length equal to 'num_words' with index corresponding to <eos>.
   * Convert the input string to a vector of indices that represent
   * the words in the input string.
   */
  std::vector<float> index_vector(num_words, GetIndexForWord("<eos>"));
  int num_words = ConverToIndexVector(input_text, &index_vector);

  executor->arg_dict()["data0"].SyncCopyFromCPU(index_vector.data(), index_vector.size());
  executor->arg_dict()["data1"] = num_words;

  // Run the forward pass.
  executor->Forward(false);

  /*
   * The output is available in executor->outputs. It is a vector of
   * NDArray. We need to find the index in that vector that
   * corresponds to the output symbol "sentimentnet0_hybridsequential0_dense0_fwd_output".
   */
  const std::string output_symbol_name = "sentimentnet0_hybridsequential0_dense0_fwd_output";
  int output_index = GetIndexForOutputSymbolName(output_symbol_name);
  std::vector<NDArray> outputs = executor->outputs;
  auto arrayout = executor->outputs[output_index].Copy(global_ctx);
  /*
   * We will run sigmoid operator to find out the sentiment score between
   * 0 and 1 where 1 represents positive.
   */
  NDArray ret;
  Operator("sigmoid")(arrayout).Invoke(ret);
  ret.WaitToRead();

  return ret.At(0, 0);
}


/*
 * The destructor frees the executor and notifies MXNetEngine to shutdown.
 */
Predictor::~Predictor() {
  if (executor) {
    delete executor;
  }
  MXNotifyShutdown();
}


/*
 * The function prints the usage information.
 */
void printUsage() {
    std::cout << "Usage:" << std::endl;
    std::cout << "simple_rnn " << std::endl
              << "--input Input movie review line."
              << "e.g. \"This movie is the best\""  << std::endl
              << "[--max_num_words]  "
              << "The number of words in the sentence to be considered for sentiment analysis. "
              << "Default is " << DEFAULT_NUM_WORDS << std::endl
              << "[--gpu]  Specify this option if workflow needs to be run in gpu context "
              << std::endl;
}


/*
 * The function downloads the model files from s3 bucket.
 */
void Download_files(const std::vector<std::string> model_files) {
  std::string wget_command("wget -nc ");
  std::string s3_url(DEFAULT_S3_URL);
  for (auto &file : model_files) {
    std::ostringstream oss;
    oss << wget_command << s3_url << file << " -O " << file;
    int status = system(oss.str().c_str());
    LG << "Downloading " << file << " with status " << status;
  }
  return;
}


int main(int argc, char** argv) {
  std::string model_file_json = "./sentiment_analysis-symbol.json";
  std::string model_file_params ="./sentiment_analysis-0001.params";
  std::string input_dictionary = "./sentiment_token_to_idx.txt";
  std::string input_review = "This movie is the best";

  int num_words = DEFAULT_NUM_WORDS;
  bool use_gpu = false;

  int index = 1;
  while (index < argc) {
    if (strcmp("--input", argv[index]) == 0) {
      index++;
      input_review = (index < argc ? argv[index]:input_review);
    } else if (strcmp("--max_num_words", argv[index]) == 0) {
      index++;
      if (index < argc) {
        std::istringstream(std::string(argv[index])) >> num_words;
      }
    } else if (strcmp("--gpu", argv[index]) == 0) {
      use_gpu = true;
    } else if (strcmp("--help", argv[index]) == 0) {
      printUsage();
      return 0;
    }
    index++;
  }


  /*
   * Download the trained RNN model file, param file and dictionary file.
   * The dictionary file contains word to index mapping.
   * Each line of the dictionary file contains a word and the unique index for that word separated
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
    Predictor predict(model_file_json, model_file_params, input_dictionary, use_gpu,
                      num_words);

    // Run the forward pass to predict the sentiment score.
    float sentiment_score = predict.PredictSentiment(input_review);
    LG << "The sentiment score between 0 and 1, (1 being positive)=" << sentiment_score;
  } catch (std::runtime_error &error) {
    LG << MXGetLastError();
    LG << "Execution failed with ERROR: " << error.what();
    return 1;
  } catch (...) {
    /*
     * If underlying MXNet code has thrown an exception the error message is
     * accessible through MXGetLastError() function.
     */
    LG << "Execution failed with following MXNet error";
    LG << MXGetLastError();
    return 1;
  }
  return 0;
}
