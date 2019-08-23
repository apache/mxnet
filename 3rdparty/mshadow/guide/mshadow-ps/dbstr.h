#pragma once
#include <mshadow/tensor.h>
#include <sstream>

template<typename DType>
std::string dbstr(mshadow::Tensor<mshadow::cpu, 1, DType> ts) {
  std::stringstream ss;
  for (mshadow::index_t i = 0; i < ts.size(0); ++i)
    ss << ts[i] << " ";
  ss << "\n";
  return ss.str();
}

template<typename DType>
std::string dbstr(mshadow::Tensor<mshadow::cpu, 2, DType> ts) {
  std::stringstream ss;
  for (mshadow::index_t i = 0; i < ts.size(0); ++i) {
    for (mshadow::index_t j = 0; j < ts.size(1); ++j) {
      ss << ts[i][j] << " ";
    }
    ss << "\n";
  }
  ss << "\n";
  return ss.str();
}

template<typename DType>
std::string dbstr(mshadow::Tensor<mshadow::cpu, 3, DType> ts) {
  std::stringstream ss;
  for (mshadow::index_t i = 0; i < ts.size(0); ++i) {
    ss << dbstr(ts[i]) << "\n";
  }
  ss << "\n";
  return ss.str();
}
