#pragma once

#include <iostream>
#include "layer/layer.h"

/*! \brief a demonstration of how to define a new layer */
class DummyLayer : public Layer {
  // dummy layer that adds delta to matrix when ff, and substracts when bp
public:
  DummyLayer(FloatT delta_) : delta(delta_) {}
  virtual void forward_cpu(
    const std::vector<Blob> & inputs,
    const std::vector<Blob> & outputs) {
    std::cout << "ff cpu" << std::endl;
  }
  virtual void forward_gpu(
    const std::vector<Blob> & inputs,
    const std::vector<Blob> & outputs) {
    std::cout << "ff gpu" << std::endl;
  }
  virtual void backward_cpu(
    const std::vector<Blob> & inputs,
    const std::vector<Blob> & outputs,
    const std::vector<bool> & propagate_down = {}) {
    std::cout << "bp cpu" << std::endl;
  }
  virtual void backward_gpu(
    const std::vector<Blob> & inputs,
    const std::vector<Blob> & outputs,
    const std::vector<bool> & propagate_down = {}) {
    std::cout << "bp gpu" << std::endl;
  }
private:
  FloatT delta;
};

/*! \brief a demonstration of how to define a new layer with narray interface */
class DummyNArrayLayer : public LayerWithNArrayInterface {
public:
  virtual void forward(const std::vector<NArray> & inputs,
    const std::vector<NArray *> & outputs) {
    *outputs[0] = *outputs[0] + inputs[0];
  }
  virtual void backward(const std::vector<NArray> & inputs,
    const std::vector<NArray *> & outputs,
    const std::vector<bool> & propagate_down = {}) {
    *outputs[0] = *outputs[0] + inputs[0];
  }
};
