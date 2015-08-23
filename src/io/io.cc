// Copyright (c) 2015 by Contributors
#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE

#include <mxnet/io.h>
#include <dmlc/logging.h>
#include <dmlc/config.h>
#include <mshadow/tensor.h>
#include <string>
#include <vector>
#include <fstream>
#include "iter_mnist-inl.h"
#include "../utils/random.h"

namespace mxnet {
  IIterator<DataBatch> *CreateIterator(
          const std::vector< std::pair<std::string, std::string> > &cfg) {
    size_t i = 0;
    IIterator<DataBatch> *it = NULL;
    for (; i < cfg.size(); ++i) {
      const char *name = cfg[i].first.c_str();
      const char *val  = cfg[i].second.c_str();
      if (!strcmp(name, "iter")) {
        if (!strcmp(val, "mnist")) {
            CHECK(it == NULL) << "mnist cannot chain over other iterator";
            it = new MNISTIterator(); continue;
        }
        CHECK(!strcmp(val, "mnist")) << "Currently only have mnist iterator";
      }
      if (it != NULL) {
        it->SetParam(name, val);
      }
    }
    CHECK(it != NULL) << "must specify iterator by iter=itername";
    return it;
  }

  IIterator<DataBatch> *CreateIteratorFromConfig(const char* cfg_path) {
    std::ifstream ifs(cfg_path, std::ifstream::in);
    std::vector< std::pair< std::string, std::string> > itcfg;
    dmlc::Config cfg(ifs);
    for (dmlc::Config::ConfigIterator iter = cfg.begin(); iter != cfg.end(); ++iter) {
        dmlc::Config::ConfigEntry ent = *iter;
      itcfg.push_back(std::make_pair(ent.first, ent.second));
    }
    // Get the data and init
    return CreateIterator(itcfg);
  }

  IIterator<DataBatch> *CreateIteratorByName(const char* iter_name) {
    IIterator<DataBatch> *it = NULL;
    // Currently only support mnist
    if (!strcmp(iter_name, "mnist")) {
      CHECK(it == NULL) << "mnist cannot chain over other iterator";
      it = new MNISTIterator();
    }
    CHECK(!strcmp(iter_name, "mnist")) << "Currently only have mnist iterator";
    return it;
  }
}  // namespace mxnet
