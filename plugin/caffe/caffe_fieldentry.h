/*!
 * Copyright (c) 2016 by Contributors
 * \file caffe_fieldentry.h
 * \brief Implement FieldEntry<caffe::LayerParameter>
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

  /**
   * /brief Customize set method for LayerParameter
   * /tparam value string of caffe's layer configuration
   * */
  virtual void Set(void *head, const std::string &value) const {
    NetParameter net_param;
    if (!ReadProtoFromTextContent(value, &net_param))
      CHECK(false)<< "Caffe Net Prototxt: " << value << "Initialized Failed";

    CHECK_EQ(net_param.layer_size(), 1) << "Protoxt " << value <<" is more than one layer";
    LayerParameter *layer_param = new LayerParameter(net_param.layer(0));
    this->Get(head) = (*layer_param);
  }

  virtual void PrintValue(std::ostream &os, LayerParameter value) const { // NOLINT(*)
  }

  virtual void PrintDefaultValueString(std::ostream &os) const {  // NOLINT(*)
    os << '\'' << default_value_.name().c_str() << '\'';
  }

  // override set_default
  inline FieldEntry<LayerParameter> &set_default(const std::string &value) {
    NetParameter net_param;
    if (!ReadProtoFromTextContent(value, &net_param))
      CHECK(false)<< "Caffe Net Prototxt: " << value << "Initialized Failed";

    CHECK_EQ(net_param.layer_size(), 1) << "Protoxt " << value <<" is more than one layer";
    default_value_ = LayerParameter(net_param.layer(0));
    has_default_ = true;
    // return self to allow chaining
    return this->self();
  }
};

}  // namespace parameter
}  // namespace dmlc

#endif  // PLUGIN_CAFFE_CAFFE_FIELDENTRY_H_
