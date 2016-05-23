#include<caffe/common.hpp>
#include<mshadow/tensor.h>
#include<dmlc/logging.h>

namespace mxnet{
namespace op{

class CaffeMode{
  public:
  template<typename xpu> static void SetMode();
};

}
}
