#include"layerparameter_stream.h"

namespace dmlc{
  namespace parameter{
    std::istringstream &operator>>(std::istringstream &is, ::caffe::LayerParameter &para_){
	  	return is;
    }
    std::ostream &operator<<(std::ostream &os, ::caffe::LayerParameter &para_){
		  return os;
    }
  }
}
