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

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <thread>
#include <iomanip>
#include <mxnet/c_predict_api.h>

// Read file to buffer
class BufferFile {
 public :
  std::string file_path_;
  std::size_t length_ = 0;
  std::unique_ptr<char[]> buffer_;

  explicit BufferFile(const std::string& file_path)
    : file_path_(file_path) {

    std::ifstream ifs(file_path.c_str(), std::ios::in | std::ios::binary);
    if (!ifs) {
      std::cerr << "Can't open the file. Please check " << file_path << ". \n";
      return;
    }

    ifs.seekg(0, std::ios::end);
    length_ = static_cast<std::size_t>(ifs.tellg());
    ifs.seekg(0, std::ios::beg);
    std::cout << file_path.c_str() << " ... " << length_ << " bytes\n";

    buffer_.reset(new char[length_ + 1]);
    buffer_[length_] = '\0';
    ifs.read(buffer_.get(), length_);
    ifs.close();
  }

  std::size_t GetLength() {
    return length_;
  }

  char* GetBuffer() {
    return buffer_.get();
  }
};

int main(int argc, char* argv[]) {

  // Models path for your model, you have to modify it
  std::string json_file = "Inception-BN-symbol.json";
  std::string param_file = "Inception-BN-0126.params";

  BufferFile json_data(json_file);
  BufferFile param_data(param_file);


  // Parameters
  int dev_type = 1;  // 1: cpu, 2: gpu
  int dev_id = 0;  // arbitrary.
  mx_uint num_input_nodes = 1;  // 1 for feedforward
  const char* input_key[1] = { "data" };
  const char** input_keys = input_key;

  // Image size and channels
  const size_t width = 224;
  const size_t height = 224;
  const size_t channels = 3;

  const mx_uint input_shape_indptr[2] = { 0, 4 };
  const mx_uint input_shape_data[4] = { 1,
                                        static_cast<mx_uint>(channels),
                                        static_cast<mx_uint>(height),
                                        static_cast<mx_uint>(width) };

  if (json_data.GetLength() == 0 || param_data.GetLength() == 0) {
    return EXIT_FAILURE;
  }

  auto image_size = width * height * channels;

  std::vector<mx_float> image_data(image_size);
  for (int i = 0; i < image_size; i++) {
    image_data[i] = 0;
  }

  for (int i = 0; i < 1234; i++) {
      // Create Predictor
      PredictorHandle pred_hnd;
      MXPredCreate(static_cast<const char*>(json_data.GetBuffer()),
                   static_cast<const char*>(param_data.GetBuffer()),
                   static_cast<int>(param_data.GetLength()),
                   dev_type,
                   dev_id,
                   num_input_nodes,
                   input_keys,
                   input_shape_indptr,
                   input_shape_data,
                   &pred_hnd);

      // Set Input Image
      MXPredSetInput(pred_hnd, "data", image_data.data(), static_cast<mx_uint>(image_size));

      // Do Predict Forward
      MXPredForward(pred_hnd);

      mx_uint output_index = 0;

      mx_uint* shape = nullptr;
      mx_uint shape_len;

      // Get Output Result
      MXPredGetOutputShape(pred_hnd, output_index, &shape, &shape_len);

      std::size_t size = 1;
      for (mx_uint i = 0; i < shape_len; ++i) { size *= shape[i]; }

      std::vector<float> data(size);

      MXPredGetOutput(pred_hnd, output_index, &(data[0]), static_cast<mx_uint>(size));

      // Release Predictor
      MXPredFree(pred_hnd);
  }

  printf("run successfully\n");

  return EXIT_SUCCESS;
}
