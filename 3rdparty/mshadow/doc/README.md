MShadow Documentation
=====
This is the documentation for mshadow: A Lightweight CPU/GPU Matrix/Tensor Template Library in C++/CUDA. 

### Links to Topics

* [Tutorial](../guide)
* [API Documentation](http://homes.cs.washington.edu/~tqchen/mshadow/doc)
  - You can run ```./mkdoc.sh``` to make the document locally
* [Tutorial about Expression Template](../guide/exp-template)
* [Writing Multi-GPU and Distributed ML](../guide/mshadow-ps)
* [Compile Configuration script](../make)
* [Expression API](#expression-api)
  - Expression api introduces the concept of expression in mshadow

Expression API
=====
Expression is the key concept in mshadow, a common operation of mshadow is ```tensor = some code to construct expression```

There are three major types of expression:
* Mapper expression: only contain element-wise operations of Mapper expressions  
  - Mapper expression can used as composition component of other operations.
  - Tensor, scalar are Mapper expressions
  - Example: ``` weight =  - eta * (grad + lambda * weight)```  is a Mapper expression.
  - Mapper expressions are translated using expression template code implemented by mshadow.
  - ***Assign safety***: Element-wise mapping are assign safe, which means, we can write ```A = A * 2 + B```, making lvalue appear in expression, the results are still correct.
* Chainer expression: may contain element-wise operation such as reduction and broadcast
  - Example: ```dst = mirror(src)``` is a chainer expression
  - ***Assign safety***: Most of the chainer extensions are not assignment safe, which means user should avoid putting target in source epression.
* Complex expression: complex operations, need special translation rule to translate to specific implementations.
   - Complex expression can not be used as composition component of other operations.
   - Example: ``` dot(lhs.T(), rhs)```,  is complex expression, we can not write
``` dst =  1.0 + dot(lhs.T(), rhs)```
   - But limited syntax is supported depending on specification, for example, we do support ``` dst +=  2.0f * dot(lhs.T(), rhs)```
   - Complex expressions are translated into specific implementations such as BLAS.

### Element-wise Operations
The basic binary operators are overloaded to composite Mapper expressions, so we can write 
```c++
weight = (-eta) * (grad + lambda * weight);
```
We can also use customized binary operators, and unary operators:
```c++
struct maximum {
  MSHADOW_XINLINE static float Map(float a, float b) {
    return a > b ? a : b;
  }
};
template<typename xpu>
void ExampleMaximum(Tensor<xpu, 2> out,
                    const Tensor<xpu, 2> &A,
                    const Tensor<xpu, 2> &B) {
  out= 10.0f * F<maximum>(A+1.0f, B); 
}
struct sigmoid {
  MSHADOW_XINLINE static float Map(float a) {
    return 1.0f/(1.0f+expf(-a));
  }
};
template<typename xpu>
void ExampleSigmoid(Tensor<xpu, 2> out, const Tensor<xpu, 2> &in) {
  // equivalent to out = sigmoid(in*2) + 1; 
  out = F<op::plus>(F<sigmoid>(in * 2.0f), ScalarExp(1.0f));
}
```
### Matrix Multiplications
Matrix multiplications are supported by following syntax, with things brackets [] are optional
```
dst <sv> [scale*] dot(lhs [.T()] , rhs [.T()]), <sv> can be =,+=,-=
```
Example:
```c++
template<typename xpu>
void Backprop(Tensor<xpu, 2> gradin,
              const Tensor<xpu, 2> &gradout,
              const Tensor<xpu, 2> &netweight) {
  gradin = 2.0 * dot(gradout, netweight.T());
}
```

### Introducing Expression Extensions
Naming conventions:
* ```Tensor<xpu, dim>``` to refer to any Tensor with device any device and dimension. 
* ```xpu```, ```dim```, are implicit template parameters. 
* ```Expr<xpu, dim>``` will be used to refer to any mapper expression with type ```Tensor<xpu,dim>```.

List of functions:
* [reshape](#reshape): reshapes a tensor to another shape, number of content must be same
* [broadcast<?>](#broadcast): replicate a 1 dimension tensor in certain dimension
* [repmat](#repmat), special case of broadcast<0>: repeat vector over rows to form a matrix
* [sumall_except_dim<?>](#sumall_except_dim): sum over all the dimensions, except the dimension specified in template parameter
* [sum_rows](#sum_rows): special case of sumall_except_dim<0>, sum of rows in the matrix
* [unpack_patch2col](#unpack_patch2col): unpack local (overlap) patches of image to column of mat, can be used to implement convolution
* [pack_col2patch](#pack_col2patch): reverse operation of unpack_patch2col, can be used to implement deconvolution
* [pool](#pool): do pooling on image
* [unpool](#unpool): get gradient of pooling result
* [crop](#crop): crop the original image to a smaller size
* [mirror](#mirror): get the mirrored result of input expression

======
##### reshape
* ```reshape(Expr<xpu,dim> src, Shape<dimdst> oshape)```
* reshapes a tensor to another shape, total number of elements must be same
* parameters:
  - src:  input data
  - oshape: target shape
* result expression type: ```Tensor<xpu, dimdst>``` with ```shape=oshape```, is Mapper expression
```c++
void ExampleReshape(void) {
  Tensor<cpu, 2> dst = NewTensor<cpu>(Shape2(4, 5));
  Tensor<cpu, 1> src = NewTensor<cpu>(Shape1(20), 1.0f); 
  dst = reshape(src, dst.shape_);
  ...
}
```
======

##### broadcast
* ```broadcast<dimcast>(Tensor<xpu,1> src, Shape<dimdst> oshape)```
* replicate a 1 dimension tensor certain dimension, specified by template parameter dimcast
* parameters:
  - src: input 1 dimensional tensor
  - oshape: shape of output
* return expression type: ```Tensor<xpu, dimdst>```, ```shape = oshape```, is Chainer expression 
```c++
void ExampleBroadcast(void) {
  Tensor<cpu, 2> dst = NewTensor<cpu>(Shape2(2, 3));
  Tensor<cpu, 1> src = NewTensor<cpu>(Shape1(2), 1.0f);
  src[0] = 2.0f; src[1] = 1.0f;
  dst = broadcast<0>(src, dst.shape_);
  // dst[0][0] = 2, dst[0][1] = 2; dst[1][0]=1, dst[1][1] = 1
  ...
}
```
======
##### repmat
* ```repmat(Tensor<xpu, 1> src, int nrows) ```
* special case of broadcast, repeat 1d tensor over rows
* input parameters:
  - src: input vector
  - nrows: number of rows in target
* return expression type:  ```Tensor<xpu, 2>```, with ```shape=(nrows, src.size(0))```,  is Chainer expression
```c++
void ExampleRepmat(void) {
  Tensor<cpu,2> dst = NewTensor<cpu>(Shape2(3, 2));
  Tensor<cpu,1> src = NewTensor<cpu>(Shape1(2), 1.0f);
  src[0] = 2.0f; src[1] = 1.0f;
  dst = repmat(src, 3);
  // dst[0][0] = 2, dst[0][1] = 1; dst[1][0]=2, dst[1][1] = 1
  ...
}
```
======
##### sumall_except_dim
* ```sumall_except_dim<dimkeep>(Expr<xpu,dim> src) ```
* sum over all dimensions, except dimkeep
* input parameters:
  - src: input mapper expression
* return expression type:  ```Tensor<xpu, 1>```, with ```shape=(src.size(dimkeep))```,  is Complex expression
* Syntax: ```dst [sv] [scale*] sumall_except_dim<dimkeep>(src) , <sv> can be =, +=, -=, *=, /=````
```c++
void ExampleSumAllExceptDim(void) {
  Tensor<cpu,3> src = NewTensor<cpu>(Shape3(2, 3, 2), 1.0f);
  Tensor<cpu,1> dst = NewTensor<cpu>(Shape1(3), 1.0f);
  dst += sum_all_except<1>(src * 2.0f);
  // dst[0] = 1.0 + 4.0 *2.0 = 9.0
  ...
}
```
======
##### sum_rows
* ```sum_rows(Expr<xpu, 2> src) ```
* sum of rows in the matrix
* input parameters:
  - src: input mapper  expression
* return expression type:  ```Tensor<xpu,1>```, with ```shape=(src.size(0))```,  is Complex expression
* Syntax: ```dst [sv] [scale*] sum_rows(src) , <sv> can be =,+=,-=,*=,/=````
```c++
void ExampleSumRows(void) {
  Tensor<cpu, 2> src = NewTensor<cpu>(Shape2(3, 2), 1.0f);
  Tensor<cpu, 1> dst = NewTensor<cpu>(Shape1(2), 1.0f);
  dst += sum_rows(src + 1.0f);
  // dst[0] = 1.0 + 3.0 *(1.0+1.0) = 7.0
  ...
}
```
======
##### unpack_patch2col
* ```unpack_patch2col(Expr<xpu,3> img, int psize_y, int p_size_x, int pstride) ```
* unpack local (overlap) patches of image to column of mat, can be used to implement convolution, after getting unpacked mat, we can use: ```output = dot(weight, mat)``` to get covolved results, the relations:
  - weight; shape[0]: out_channel, shape[1]: ichannel * psize_y * psize_x
  - output; shape[0]: out_channel, shape[1]: out_height * out_width * num_of_images
  -  out_height = (in_height - psize_y) / pstride + 1, this means we pad inperfect patch with 0
  - out_width  = (in_width - psize_x) / pstride + 1
* input parameters:
  - img: source image, can be expression; (in_channels, in_height, in_width)
  - psize_y height of each patch
  - psize_x width of each patch
  - pstride: stride of each patch
* return expression type:  ```Tensor<xpu, 2>```, with ```shape=(in_channel*psize_x*psize_y, out_height*out_width)```,  is Chainer expression
```c++
void ExampleCovolution(Tensor<cpu, 3> dst, Tensor<cpu, 3> src,
                       Tensor<cpu, 2> weight, int ksize, int stride) {
  int o_height = (src.size(1)- ksize) / stride + 1;
  int o_width  = (src.size(2)- ksize) / stride + 1;
  utils::Assert(weight.size(1) == src.size(0) * ksize * ksize);
  TensorContainer<cpu, 2> tmp_col(Shape2(src.size(0) * ksize * ksize,
                                         o_height * o_width)); 
  TensorContainer<cpu, 2> tmp_dst(Shape2(weight.size(0),
                                         o_height * o_width)); 
  tmp_col = unpack_patch2col(src, ksize, ksize, stride);
  tmp_dst = dot(weight, tmp_col);
  dst = reshape(tmp_dst, dst.shape_);
}
```

======
##### pack_col2patch
* ```pack_col2patch(Tensor<xpu, 2> mat, Shape<3> imshape, int psize_y, int psize_x, int pstride) ````
* reverse operation of unpack_patch2col, can be used to implement deconvolution
* input parameters:
  - mat: source mat, same shape as output of unpack_patch2col
  - imshape: shape of target image
  - psize_y height of each patch
  - psize_x width of each patch
  - pstride: stride of each patch
* return expression type:  ```Tensor<xpu, 3>```, with ```shape = imshape```,  is Chainer expression
```c++
void ExampleDecovolution(Tensor<cpu, 3> bottom, Tensor<cpu, 3> top,
                         Tensor<cpu, 2> weight, int ksize, int stride) {
  int o_height = (bottom.size(1)- ksize) / stride + 1;
  int o_width  = (bottom.size(2)- ksize) / stride + 1;
  utils::Assert(weight.size(1) == bottom.size(0) * ksize * ksize);
  TensorContainer<cpu, 2> tmp_col(Shape2(bottom.size(0) * ksize * ksize,
                                         o_height * o_width)); 
  TensorContainer<cpu, 2> tmp_dst(Shape2(weight.size(0), o_height*o_width)); 
  tmp_dst = reshape(top, tmp_dst.shape_);
  tmp_col = dot(weight.T(), tmp_dst);
  bottom = pack_col2patch(tmp_col, bottom.shape_, ksize, ksize, stride);
}
```

======
##### pool
* ```pool<Reducer>(Expr<xpu, dim> img, [Shape<2> pshape,] int ksize_y, int ksize_x, int kstride)```
* Pooling on image with specify kernel size and stride, can be used to implement max pooilng and other pooling layer
* input parameters:
  - Reducer: operation can be max or sum
  - img: source image, can be expression; (in_channels, in_height, in_width)
  - [optional] Shape<2> pshape, output shape
  - ksize_y height of each patch
  - ksize_x width of each patch
  - kstride: stride of each patch
* return expression:  ```Expr<xpu, dim>```, with ```shape = (in_channel, (out_height - ksize) / kstride + 1, (out_width - ksize) / kstride + 1)```, or expression in pshape
  - Chainer expression
```c++
void ExampleMaxPooling(TensorContainer<cpu, 3> &data, int ksize, int stride) {
  TensorContainer<cpu, 3> pooled(Shape3(data.size(0),
                                        (data.size(2) - ksize) / kstride + 1), 
                                        (data.size(1) - ksize) / kstride + 1));
  pooled = pool<red::maximum>(data, ksize, ksize, stride);
}
```

======
##### unpool
* ```unpool<Reducer>(Tensor<xpu, 4> data_src, Tensor<xpu, 4> data_pooled, Tensor<xpu, 4> grad_pooled, int ksize_y,  int ksize_x, int kstride)```
* Unpooling on image with specify kernel size and stride, can be used to implement backprop of max pooilng and other pooling layer
* input parameters:
  - Reducer: operation can be max or sum
  - data_src: source image batch. 
  - data_pooled: pooled image batch. 
  - grad_pooled: gradient of upper layer
  - ksize_y height of each patch
  - ksize_x width of each patch
  - kstride: stride of each patch
* return:
  Expression, same shape to data_src
```c++
void ExampleMaxUnpooling(Tensor<cpu, 4> &data_src, Tensor<cpu, 4> &data_pooled, 
                         Tensor<cpu, 4> &grad_pooled, int ksize, int kstride) {
  TensorContainer<cpu, 4> grad(data_src.shape_);
  grad = unpool<red::maximum>(data_src, data_pooled,
                              grad_pooled, ksize, ksize, kstride);
}
```

======
##### crop
* ```crop(Expr<xpu, dim> src, Shape<2> oshape, int start_height, int start_width)```
* input parameters:
 - src: input expression 
 - oshape: output shape after crop
 - start_height: start height for cropping
 - start_width: start width for cropping
* Can also be ```crop(Expr<xpu, dim> src, Shape<2> oshape)``` where the crop will happen in center. 
* return
 - cropped expression
```c++
void ExampleCrop(TensorContainer<cpu, 3> img, int start_height, int start_width) {
  TensorContainer<cpu> cropped(Shape3(img.size(0),
                                      img.size(1) - start_height,
                                      img.size(2) - start_width));
  cropped = crop(img, start_height, start_width);
}
```

======
##### mirror
* ```mirrow(Expr<xpu, dim> src)```
* input:
    - src, source expression to be mirrored
* output:
    - expression of mirrored result
```c++
void ExampleMirror(TensorContainer<cpu, 3> img) {
  TensorContainer<cpu> mirrored(img.shape_);
  mirrored = mirror(img);
}
```

