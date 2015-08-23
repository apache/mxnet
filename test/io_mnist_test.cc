// Copyright (c) 2015 by Contributors
// IO test code

#include <dmlc/io.h>
#include <dmlc/logging.h>
#include <dmlc/config.h>
#include <iostream>
#include <fstream>
#include "mxnet/io.h"
#include "../src/io/iter_mnist-inl.h"

using namespace std;
using namespace mxnet;
using namespace dmlc;

void InitIter(IIterator<DataBatch>* itr,
        const std::vector< std::pair< std::string, std::string> > &defcfg) {
    for (size_t i = 0; i < defcfg.size(); ++i) {
      itr->SetParam(defcfg[i].first.c_str(), defcfg[i].second.c_str());
    }
    itr->Init();
}

IIterator<DataBatch>* CreateIterators(
        const std::vector< std::pair< std::string, std::string> >& cfg) {
    IIterator<DataBatch>* data_itr = NULL;
    int flag = 0;
    std::string evname;
    std::vector< std::pair< std::string, std::string> > itcfg;
    std::vector< std::pair< std::string, std::string> > defcfg;
    for (size_t i = 0; i < cfg.size(); ++i) {
      const char *name = cfg[i].first.c_str();
      const char *val  = cfg[i].second.c_str();
      if (!strcmp(name, "data")) {
          flag = 1; continue;
      }
      if (!strcmp(name, "eval")) {
          flag = 2; continue;
      }
      if (!strcmp(name, "pred")) {
          flag = 3; continue;
      }
      if (!strcmp(name, "iterend") && !strcmp(val, "true")) {
          if (flag == 1) {
              data_itr = mxnet::CreateIterator(itcfg);
          }
          flag = 0; itcfg.clear();
      }
      if (flag == 0) {
          defcfg.push_back(cfg[i]);
      } else {
          itcfg.push_back(cfg[i]);
      }
    }
    if (data_itr != NULL) {
        InitIter(data_itr, defcfg);
    }
    return data_itr;
}

/*!
 *  Usage: ./io_mnist_test /path/to/io_config/file
 *  Example	
 *  data = train
 *  iter = mnist
 *      path_img = "./data/mnist/train-images-idx3-ubyte"
 *      path_label = "./data/mnist/train-labels-idx1-ubyte"
 *      shuffle = 1
 *  iterend = true
 *  input_shape = 1,1,784
 *  batch_size = 100
 *	
 */

int main(int argc, char** argv) {
  std::ifstream ifs(argv[1], std::ifstream::in);
  std::vector< std::pair< std::string, std::string> > itcfg;
  Config cfg(ifs);
  for (Config::ConfigIterator iter = cfg.begin(); iter != cfg.end(); ++iter) {
    Config::ConfigEntry ent = *iter;
    itcfg.push_back(std::make_pair(ent.first, ent.second));
  }
  // Get the data and init
  IIterator<DataBatch>* data_itr = CreateIterators(itcfg);
  data_itr->BeforeFirst();
  int batch_dir = 0;
  while (data_itr->Next()) {
      std::cout  << "Label of Batch " << batch_dir++ << std::endl;
      // print label
      DataBatch db = data_itr->Value();
      mshadow::Tensor<mshadow::cpu, 2> label = db.data[1].get<mshadow::cpu, 2, float>();
      for (size_t i = 0; i < label.shape_.shape_[0]; i++)
          std::cout << label.dptr_[i] << " ";
      std::cout << "\n";
  }
}
