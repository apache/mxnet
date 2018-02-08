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

#include <cstdlib>
#include <cstdio>
#include <string>
#include <iostream>
#include <thread>
#include <unordered_map>
#include "storage/cpu_shared_storage_manager.h"

enum class Mode : int {
  Producer,
  Consumer
};

const char* get_mode_name(const Mode& mode) {
  switch (mode) {
    case Mode::Producer:
      return "producer";
    case Mode::Consumer:
      return "consumer";
  }

  return "";
}

std::ostream& operator<<(std::ostream& stream, const Mode& mode) {
  switch (mode) {
    case Mode::Producer:
    case Mode::Consumer:
      stream << get_mode_name(mode);
      break;
  }

  return stream;
}

const char key[] = "LedZeppelin";
const char msg[] = "Ah-ah, ah! \n"
  "Ah-ah, ah! \n"
  "We come from the land of the ice and snow \n"
  "From the midnight sun, where the hot springs flow";
const int wait_time = 500;
const std::size_t size = 1024;
auto context = mxnet::Context::CPUShared();

int producer() {
  auto mode_name = std::string(get_mode_name(Mode::Producer)) + ": ";

  auto manager = mxnet::storage::AbstractManager::make<mxnet::storage::CPUSharedStorageManager>();

  std::cout << mode_name << "Allocating " << size << " bytes..." << std::endl;

  auto memory = manager->Allocate(key, size, context);

  if (!memory || !memory->dptr) {
    std::cout << mode_name << "Could not allocate memory" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << mode_name << "Memory allocated" << std::endl;

  std::memcpy(memory->dptr, msg, sizeof(msg));

  std::cout << mode_name << "Message written, waiting " << wait_time << "ms..." << std::endl;

  std::this_thread::sleep_for(std::chrono::milliseconds(wait_time));

  std::cout << mode_name << "Removing memory reference..." << std::endl;
  memory.reset();
  std::cout << mode_name << "Done" << std::endl;

  return EXIT_SUCCESS;
}

int consumer() {
  auto mode_name = std::string(get_mode_name(Mode::Consumer)) + ": ";

  auto manager = mxnet::storage::AbstractManager::make<mxnet::storage::CPUSharedStorageManager>();

  std::cout << mode_name << "Attaching to shared memory with " << key << " key..." << std::endl;

  std::shared_ptr<mxnet::storage::Handle> memory;

  for (int i = 0; i < 10 && !memory; ++i) {
    try {
      memory = manager->Attach(key);
    } catch (std::exception& /*e*/) {
      std::cout << mode_name << "Retrying..." << std::endl;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(wait_time / 10));
  }

  if (!memory) {
    std::cout << mode_name << "Attaching memory failed" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << mode_name << "Memory attached, reading message..." << std::endl;

  auto read_msg = static_cast<const char*>(memory->dptr);
  std::cout << mode_name << "Message: " << read_msg << std::endl;

  if (std::string(msg) != read_msg) {
    std::cout << mode_name << "Message is not equal to the written!" << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cout << "Please provide the mode in which to call the helper as argument: " << std::endl
              << "[" << Mode::Producer << ", " << Mode::Consumer << "]" << std::endl;
    return EXIT_FAILURE;
  }

  auto mode = std::string(argv[1]);

  std::cout << "Mode provided: " << mode << std::endl;

  if (mode == get_mode_name(Mode::Producer)) {
    return producer();
  }

  if (mode == get_mode_name(Mode::Consumer)) {
    return consumer();
  }

  std::cout << "Unknown mode: " << mode << std::endl;

  return EXIT_SUCCESS;
}
