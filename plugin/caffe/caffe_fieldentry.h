/*!
 * Copyright (c) 2016 by Contributors
 * \file caffe_fieldentry.h
 * \brief FieldEntry<caffe::LayerParameter>
 * \author Haoran Wang 
 */
#ifndef PLUGIN_CAFFE_CAFFE_FIELDENTRY_H_
#define PLUGIN_CAFFE_CAFFE_FIELDENTRY_H_

#include <caffe/proto/caffe.pb.h>
#include <caffe/util/io.hpp>
#include <dmlc/parameter.h>
#include <dmlc/base.h>
#include <dmlc/json.h>
#include <dmlc/logging.h>
#include <dmlc/type_traits.h>
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>
#include <cstddef>
#include <cstdlib>
#include <sstream>
#include <limits>
#include <map>
#include <set>
#include <typeinfo>
#include <string>
#include <vector>
#include <algorithm>
#include <utility>
#include <iostream>

namespace dmlc {
namespace parameter {

using caffe::LayerParameter;
using caffe::NetParameter;
using ::google::protobuf::Message;

// specialize define for Layer Parameter
template<>
class FieldEntry<LayerParameter>
    : public FieldEntryBase<FieldEntry<LayerParameter>, LayerParameter> {
 public:
  // parent class
  typedef FieldEntryBase<FieldEntry<LayerParameter>, LayerParameter> Parent;


  bool ReadProtoFromTextContent(const std::string& text, Message* proto) const {
    bool success = google::protobuf::TextFormat::ParseFromString(text, proto);
    return success;
  }
  // override set
  // value is file position
  virtual void Set(void *head, const std::string &value) const {
    NetParameter net_param;
    /*
		std::cout << "Caffe File Name is: " << value << std::endl;
    if (!::caffe::ReadProtoFromTextFile(value, &net_param)){
		  std::cout << "Read Caffe Net Failed" << value << std::endl;
			CHECK(false);
		}
    */
    std::cout << "Caffe Content is: " << value << std::endl;
    if (!ReadProtoFromTextContent(value, &net_param)) {
      std::cout << "Caffe Net Content Failed" << value << std::endl;
      CHECK(false);
    }

    // CHECK_GE(net_param.layers_size(), 1);
    CHECK_GE(net_param.layer_size(), 1);
    // ::caffe::V1LayerParameter *layer_param = new ::caffe::V1LayerParameter(net_param.layers(0));
    LayerParameter *layer_param = new LayerParameter(net_param.layer(0));
    this->Get(head) = (*layer_param);
  }

  virtual void PrintValue(std::ostream &os, LayerParameter value) const { // NOLINT(*)
  }

  // override print default
  virtual void PrintDefaultValueString(std::ostream &os) const {  // NOLINT(*)
    os << '\'' << default_value_.name().c_str() << '\'';
  }
};

}  // namespace parameter
}  // namespace dmlc

#endif  // PLUGIN_CAFFE_CAFFE_FIELDENTRY_H_
