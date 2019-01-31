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
 * 1. Load the pre-trained model.
 * 2. Load the parameters of pre-trained model.
 * 3. Load the image to be classified  in to NDArray.
 * 4. Normalize the image using the mean of images that were used for training.
 * 5. Run the forward pass and predict the input image.
 */

#include <sys/stat.h>
#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include "mxnet-cpp/MxNetCpp.h"
#include <opencv2/opencv.hpp>

using namespace mxnet::cpp;

static mx_float DEFAULT_MEAN_R = 123.675;
static mx_float DEFAULT_MEAN_G = 116.28;
static mx_float DEFAULT_MEAN_B = 103.53;
/*
 * class Predictor
 *
 * This class encapsulates the functionality to load the model, process input image and run the forward pass.
 */

class Predictor {
 public:
    Predictor() {}
    Predictor(const std::string& model_json_file,
              const std::string& model_params_file,
              const Shape& input_shape,
              bool gpu_context_type = false,
              const std::string& synset_file = "",
              const std::string& mean_image_file = "");
    void PredictImage(const std::string& image_file);
    ~Predictor();

 private:
    void LoadModel(const std::string& model_json_file);
    void LoadParameters(const std::string& model_parameters_file);
    void LoadSynset(const std::string& synset_file);
    NDArray LoadInputImage(const std::string& image_file);
    void LoadMeanImageData();
    void LoadDefaultMeanImageData();
    void NormalizeInput(const std::string& mean_image_file);
    inline bool FileExists(const std::string& name) {
        struct stat buffer;
        return (stat(name.c_str(), &buffer) == 0);
    }
    NDArray mean_img;
    std::map<std::string, NDArray> args_map;
    std::map<std::string, NDArray> aux_map;
    std::vector<std::string> output_labels;
    Symbol net;
    Executor *executor;
    Shape input_shape;
    NDArray mean_image_data;
    NDArray std_dev_image_data;
    Context global_ctx = Context::cpu();
    std::string mean_image_file;
};


/*
 * The constructor takes following parameters as input:
 * 1. model_json_file:  The model in json formatted file.
 * 2. model_params_file: File containing model parameters
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
Predictor::Predictor(const std::string& model_json_file,
                     const std::string& model_params_file,
                     const Shape& input_shape,
                     bool gpu_context_type,
                     const std::string& synset_file,
                     const std::string& mean_image_file):
                     input_shape(input_shape),
                     mean_image_file(mean_image_file) {
  if (gpu_context_type) {
    global_ctx = Context::gpu();
  }
  // Load the model
  LoadModel(model_json_file);

  // Load the model parameters.
  LoadParameters(model_params_file);

  /*
   * The data will be used to output the exact label that matches highest output of the model.
   */
  LoadSynset(synset_file);

  /*
   * Load the mean image data if specified.
   */
  if (!mean_image_file.empty()) {
    LoadMeanImageData();
  } else {
    LG << "Mean image file for normalizing the input is not provide."
       << " We will use the default mean values for R,G and B channels.";
    LoadDefaultMeanImageData();
  }

  // Create an executor after binding the model to input parameters.
  args_map["data"] = NDArray(input_shape, global_ctx, false);
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
 * The following function loads the synset file.
 * This information will be used later to report the label of input image.
 */
void Predictor::LoadSynset(const std::string& synset_file) {
  if (!FileExists(synset_file)) {
    LG << "Synset file " << synset_file << " does not exist";
    throw std::runtime_error("Synset file does not exist");
  }
  LG << "Loading the synset file.";
  std::ifstream fi(synset_file.c_str());
  if (!fi.is_open()) {
    std::cerr << "Error opening synset file " << synset_file << std::endl;
    throw std::runtime_error("Error in opening the synset file.");
  }
  std::string synset, lemma;
  while (fi >> synset) {
    getline(fi, lemma);
    output_labels.push_back(lemma);
  }
  fi.close();
}


/*
 * The following function loads the mean data from mean image file.
 * This data will be used for normalizing the image before running the forward
 * pass.
 * The output data has the same shape as that of the input image data.
 */
void Predictor::LoadMeanImageData() {
  LG << "Load the mean image data that will be used to normalize "
     << "the image before running forward pass.";
  mean_image_data = NDArray(input_shape, global_ctx, false);
  mean_image_data.SyncCopyFromCPU(
        NDArray::LoadToMap(mean_image_file)["mean_img"].GetData(),
        input_shape.Size());
}


/*
 * The following function loads the default mean values for
 * R, G and B channels into NDArray that has the same shape as that of
 * input image.
 */
void Predictor::LoadDefaultMeanImageData() {
  LG << "Loading the default mean image data";
  std::vector<float> array;
  /*resize pictures to (224, 224) according to the pretrained model*/
  int height = input_shape[2];
  int width = input_shape[3];
  int channels = input_shape[1];
  std::vector<mx_float> default_means;
  default_means.push_back(DEFAULT_MEAN_R);
  default_means.push_back(DEFAULT_MEAN_G);
  default_means.push_back(DEFAULT_MEAN_B);
  for (int c = 0; c < channels; ++c) {
    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        array.push_back(default_means[c]);
      }
    }
  }
  mean_image_data = NDArray(input_shape, global_ctx, false);
  mean_image_data.SyncCopyFromCPU(array.data(), input_shape.Size());
}


/*
 * The following function loads the input image into NDArray.
 */
NDArray Predictor::LoadInputImage(const std::string& image_file) {
  if (!FileExists(image_file)) {
    LG << "Image file " << image_file << " does not exist";
    throw std::runtime_error("Image file does not exist");
  }
  LG << "Loading the image " << image_file << std::endl;
  std::vector<float> array;
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
  NDArray image_data = NDArray(input_shape, global_ctx, false);
  image_data.SyncCopyFromCPU(array.data(), input_shape.Size());
  return image_data;
}


/*
 * The following function runs the forward pass on the model.
 * The executor is created in the constructor.
 *
 */
void Predictor::PredictImage(const std::string& image_file) {
  // Load the input image
  NDArray image_data = LoadInputImage(image_file);

  // Normalize the image
  image_data.Slice(0, 1) -= mean_image_data;

  LG << "Running the forward pass on model to predict the image";
  /*
   * The executor->arg_arrays represent the arguments to the model.
   *
   * Copying the image_data that contains the NDArray of input image
   * to the arg map of the executor. The input is stored with the key "data" in the map.
   *
   */
  image_data.CopyTo(&(executor->arg_dict()["data"]));

  // Run the forward pass.
  executor->Forward(false);

  // The output is available in executor->outputs.
  auto array = executor->outputs[0].Copy(global_ctx);

  /*
   * Find out the maximum accuracy and the index associated with that accuracy.
   * This is done by using the argmax operator on NDArray.
   */
  auto predicted = array.ArgmaxChannel();

  /*
   * Wait until all the previous write operations on the 'predicted'
   * NDArray to be complete before we read it.
   * This method guarantees that all previous write operations that pushed into the backend engine
   * for execution are actually finished.
   */
  predicted.WaitToRead();

  int best_idx = predicted.At(0, 0);
  float best_accuracy = array.At(0, best_idx);

  if (output_labels.empty()) {
    LG << "The model predicts the highest accuracy of " << best_accuracy << " at index "
       << best_idx;
  } else {
    LG << "The model predicts the input image to be a [" << output_labels[best_idx]
       << " ] with Accuracy = " << best_accuracy << std::endl;
  }
}


Predictor::~Predictor() {
  if (executor) {
    delete executor;
  }
  MXNotifyShutdown();
}


/*
 * Convert the input string of number of hidden units into the vector of integers.
 */
std::vector<index_t> getShapeDimensions(const std::string& hidden_units_string) {
    std::vector<index_t> dimensions;
    char *p_next;
    int num_unit = strtol(hidden_units_string.c_str(), &p_next, 10);
    dimensions.push_back(num_unit);
    while (*p_next) {
        num_unit = strtol(p_next, &p_next, 10);
        dimensions.push_back(num_unit);
    }
    return dimensions;
}

void printUsage() {
    std::cout << "Usage:" << std::endl;
    std::cout << "inception_inference --symbol <model symbol file in json format>  " << std::endl
              << "--params <model params file> " << std::endl
              << "--image <path to the image used for prediction> " << std::endl
              << "--synset <file containing labels for prediction> " << std::endl
              << "[--input_shape <dimensions of input image e.g \"3 224 224\">] " << std::endl
              << "[--mean <file containing mean image for normalizing the input image>] "
              << std::endl
              << "[--gpu  <Specify this option if workflow needs to be run in gpu context>]"
              << std::endl;
}

int main(int argc, char** argv) {
  std::string model_file_json;
  std::string model_file_params;
  std::string synset_file = "";
  std::string mean_image = "";
  std::string input_image = "";
  bool gpu_context_type = false;

  std::string input_shape = "3 224 224";
    int index = 1;
    while (index < argc) {
        if (strcmp("--symbol", argv[index]) == 0) {
            index++;
            model_file_json = (index < argc ? argv[index]:"");
        } else if (strcmp("--params", argv[index]) == 0) {
            index++;
            model_file_params = (index < argc ? argv[index]:"");
        } else if (strcmp("--synset", argv[index]) == 0) {
            index++;
            synset_file = (index < argc ? argv[index]:"");
        } else if (strcmp("--mean", argv[index]) == 0) {
            index++;
            mean_image = (index < argc ? argv[index]:"");
        } else if (strcmp("--image", argv[index]) == 0) {
            index++;
            input_image = (index < argc ? argv[index]:"");
        } else if (strcmp("--input_shape", argv[index]) == 0) {
            index++;
            input_shape = (index < argc ? argv[index]:input_shape);
        } else if (strcmp("--gpu", argv[index]) == 0) {
            gpu_context_type = true;
        } else if (strcmp("--help", argv[index]) == 0) {
            printUsage();
            return 0;
        }
        index++;
    }

  if (model_file_json.empty() || model_file_params.empty() || synset_file.empty()) {
    LG << "ERROR: Model details such as symbol, param and/or synset files are not specified";
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

  try {
    // Initialize the predictor object
    Predictor predict(model_file_json, model_file_params, input_data_shape, gpu_context_type,
                      synset_file, mean_image);

    // Run the forward pass to predict the image.
    predict.PredictImage(input_image);
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
