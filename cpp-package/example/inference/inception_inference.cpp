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
 * This example demonstrates image classification workflow with pre-trained models using MXNet C++ API.
 * The example performs following tasks.
 * 1. Load the pre-trained model,
 * 2. Load the parameters of pre-trained model,
 * 3. Load the image to be classified  in to NDArray.
 * 4. Normalize the image using the mean of images that were used for training.
 * 5. Run the forward pass and predict the input image.
 */

#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include "mxnet-cpp/MxNetCpp.h"
#include <opencv2/opencv.hpp>
using namespace std;
using namespace mxnet::cpp;


Context global_ctx(kCPU, 0);
#if MXNET_USE_GPU
Context global_ctx(kGPU, 0);
#endif

/*
 * class Predictor
 *
 * This class encapsulates the functionality to load the model, process input image and run the forward pass.
 */

class Predictor {
 public:
    Predictor() {}
    Predictor(const std::string model_json,
              const std::string model_params,
              const std::string& synset_file,
              const Shape& input_shape);
    void LoadModel(const std::string& model_json_file);
    void LoadParameters(const std::string& model_parameters_file);
    void LoadInputImage(const std::string& image_file);
    void LoadSynset(const std::string& synset_file);
    void NormalizeInput(const std::string& mean_image_file);
    void RunForwardPass();
    NDArray GetImageData() {return image_data;}
    ~Predictor();

 private:
    NDArray mean_img;
    map<string, NDArray> args_map;
    map<string, NDArray> aux_map;
    std::vector<std::string> output_labels;
    Symbol net;
    Executor *executor;
    Shape input_shape;
    NDArray image_data;
};


/*
 * The constructor takes following parameters as input:
 * 1. model_json:  The model in json formatted file.
 * 2. model_params: File containing model parameters
 * 3. synset_file: File containing the list of image labels
 * 4. input_shape: Shape of input data to the model. Since this class will be running one inference at a time,
 *                 the input shape is required to be in format Shape(1, number_of_channels, height, width)
 * The input image will be resized to (height x width) size before running the inference.
 * The constructor will:
 *  1. Load the model and parameter files.
 *  2. Load the synset file.
 *  3. Invoke the SimpleBind to bind the input argument to the model and create an executor.
 *
 *  The SimpleBind is expected to be invoked only once.
 */
Predictor::Predictor(const std::string model_json,
                     const std::string model_params,
                     const std::string& synset_file,
                     const Shape& input_shape):input_shape(input_shape) {
  // Load the model
  LoadModel(model_json);

  // Load the model parameters.
  LoadParameters(model_params);

  /*
   * Load the synset file containing the image labels, if provided.
   * The data will be used to output the exact label that matches highest output of the model.
   */
  if (!synset_file.empty()) {
    LoadSynset(synset_file);
  }
  // Create an executor after binding the model to input parameters.
  args_map["data"] = NDArray(input_shape, global_ctx, false);
  executor = net.SimpleBind(global_ctx, args_map, map<string, NDArray>(),
                              map<string, OpReqType>(), aux_map);
}

/*
 * The following function loads the model from json file.
 */
void Predictor::LoadModel(const std::string& model_json_file) {
  LG << "Loading the model from " << model_json_file << std::endl;
  net = Symbol::Load(model_json_file);
}


/*
 * The following function loads the model parameters.
 */
void Predictor::LoadParameters(const std::string& model_parameters_file) {
    LG << "Loading the model parameters from " << model_parameters_file << std::endl;
    map<string, NDArray> paramters;
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
 * The following function loads the synset file.
 * This information will be used later to report the label of input image.
 */
void Predictor::LoadSynset(const string& synset_file) {
  LG << "Loading the synset file.";
  std::ifstream fi(synset_file.c_str());
  if (!fi.is_open()) {
    std::cerr << "Error opening synset file " << synset_file << std::endl;
    assert(false);
  }
  std::string synset, lemma;
  while (fi >> synset) {
    getline(fi, lemma);
    output_labels.push_back(lemma);
  }
  fi.close();
}


/*
 * The following function runs the forward pass on the model.
 * The executor is created in the constructor.
 *
 */
void Predictor::RunForwardPass() {
  LG << "Running the forward pass";
  /*
   * The executor->arg_arrays represent the arguments to the model.
   *
   * The model expects the NDArray representing the image to be classified at
   * index 0.
   * Hence, the image_data that contains the NDArray of input image is copied
   * to index 0 of executor->args_arrays.
   *
   */
  int input_position_in_args = 0;
  image_data.CopyTo(&(executor->arg_arrays[input_position_in_args]));
  NDArray::WaitAll();

  // Run the forward pass.
  executor->Forward(false);

  // The output is available in executor->outputs.
  auto array = executor->outputs[0].Copy(global_ctx);
  NDArray::WaitAll();

  float best_accuracy = 0.0;
  std::size_t best_idx = 0;

  // Find out the maximum accuracy and the index associated with that accuracy.
  for (std::size_t i = 0; i < array.Size(); ++i) {
    if (array.At(0, i) > best_accuracy) {
      best_accuracy = array.At(0, i);
      best_idx = i;
    }
  }

  if (output_labels.empty()) {
    LG << "The model predicts the highest accuracy of " << best_accuracy << " at index "
       << best_idx;
  } else {
    LG << "The model predicts the input image to be a [" << output_labels[best_idx]
       << " ] with Accuracy = " << array.At(0, best_idx) << std::endl;
  }
}


Predictor::~Predictor() {
  if (executor) {
    delete executor;
  }
  MXNotifyShutdown();
}


/*
 * The following function loads the input image.
 */
void Predictor::LoadInputImage(const std::string& image_file) {
  LG << "Loading the image " << image_file << std::endl;
  vector<float> array;
  cv::Mat mat = cv::imread(image_file);
  /*resize pictures to (224, 224) according to the pretrained model*/
  int height = input_shape[2];
  int width = input_shape[3];
  int channels = input_shape[1];
  cv::resize(mat, mat, cv::Size(height, width));
  for (int c = 0; c < channels; ++c) {
    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        array.push_back(static_cast<float>(mat.data[(i * height + j) * 3 + c]));
      }
    }
  }
  image_data = NDArray(input_shape, global_ctx, false);
  image_data.SyncCopyFromCPU(array.data(), input_shape.Size());
  NDArray::WaitAll();
}


/*
 * The following function normalizes input image data by substracting the
 * mean data.
 */
void Predictor::NormalizeInput(const std::string& mean_image_file) {
  LG << "Normalizing image using " << mean_image_file;
  mean_img = NDArray(input_shape, global_ctx, false);
  mean_img.SyncCopyFromCPU(
        NDArray::LoadToMap(mean_image_file)["mean_img"].GetData(),
        input_shape.Size());
  NDArray::WaitAll();
  image_data.Slice(0, 1) -= mean_img;
  return;
}


/*
 * Convert the input string of number of hidden units into the vector of integers.
 */
std::vector<index_t> getShapeDimensions(const std::string& hidden_units_string) {
    std::vector<index_t> dimensions;
    char *pNext;
    int num_unit = strtol(hidden_units_string.c_str(), &pNext, 10);
    dimensions.push_back(num_unit);
    while (*pNext) {
        num_unit = strtol(pNext, &pNext, 10);
        dimensions.push_back(num_unit);
    }
    return dimensions;
}

void printUsage() {
    std::cout << "Usage:" << std::endl;
    std::cout << "inception_inference --symbol <model symbol file in json format>  "
              << "--params <model params file> "
              << "--image <path to the image used for prediction "
              << "[--input_shape <dimensions of input image e.g \"3 224 224\"]"
              << "[--synset file containing labels for prediction] "
              << "[--mean file containing mean image for normalizing the input image "
              << std::endl;
}

int main(int argc, char** argv) {
  string model_file_json;
  string model_file_params;
  string synset_file = "";
  string mean_image = "";
  string input_image = "";

  std::string input_shape = "3 224 224";
    int index = 1;
    while (index < argc) {
        if (strcmp("--symbol", argv[index]) == 0) {
            index++;
            model_file_json = argv[index];
        } else if (strcmp("--params", argv[index]) == 0) {
            index++;
            model_file_params = argv[index];
        } else if (strcmp("--synset", argv[index]) == 0) {
            index++;
            synset_file = argv[index];
        } else if (strcmp("--mean", argv[index]) == 0) {
            index++;
            mean_image = argv[index];
        } else if (strcmp("--image", argv[index]) == 0) {
            index++;
            input_image = argv[index];
        } else if (strcmp("--input_shape", argv[index]) == 0) {
            index++;
            input_shape = argv[index];
        } else if (strcmp("--help", argv[index]) == 0) {
            printUsage();
            return 0;
        }
        index++;
    }

  if (model_file_json.empty() || model_file_params.empty()) {
    LG << "ERROR: Model details such as symbols and/or param files are not specified";
    printUsage();
    return 1;
  }

  if (input_image.empty()) {
    LG << "ERROR: Path to the input image is not specified.";
    printUsage();
    return 1;
  }

  std::vector<index_t> input_dimensions = getShapeDimensions(input_shape);

  /*
   * Since we are running inference for 1 image, add 1 to the input_dimensions so that
   * the shape of input data for the model will be
   * {no. of images, channels, height, width}
   */
  input_dimensions.insert(input_dimensions.begin(), 1);

  Shape input_data_shape(input_dimensions);

  // Initialize the predictor object
  Predictor predict(model_file_json, model_file_params, synset_file, input_data_shape);

  // Load the input image
  predict.LoadInputImage(input_image);

  // Normalize teh image
  if (!mean_image.empty()) {
    predict.NormalizeInput(mean_image);
  }

  // Run the forward pass.
  predict.RunForwardPass();
  return 0;
}
