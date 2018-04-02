Python Howto Examples
=====================

* [Configuring Net to Get Multiple Ouputs](multiple_outputs.py)
* [Configuring Image Record Iterator](data_iter.py)
* [Monitor Intermediate Outputs in the Network](monitor_weights.py)
* Set break point in C++ code of the symbol using gdb under Linux:

	* 	Build mxnet with following values:

		 ```
		 	DEBUG=1 
		 	USE_CUDA=0 # to make sure convolution-inl.h will be used
		 	USE_CUDNN=0 # to make sure convolution-inl.h will be used
		 ```
		 
	*  run python under gdb:  ```gdb --args python debug_conv.py```
	*  in gdb set break point on particular line of the code and run execution: 

```
(gdb) break src/operator/convolution-inl.h:120
(gdb) run
Breakpoint 1, mxnet::op::ConvolutionOp<mshadow::cpu, float>::Forward (this=0x12219d0, ctx=..., in_data=std::vector of length 3, capacity 4 = {...}, req=std::vector of length 1, capacity 1 = {...}, out_data=std::vector of length 1, capacity 1 = {...},
    aux_args=std::vector of length 0, capacity 0) at src/operator/./convolution-inl.h:121
121	               data.shape_[1] / param_.num_group * param_.kernel[0] * param_.kernel[1]);
(gdb) list
116	    }
117	    Tensor<xpu, 4, DType> data = in_data[conv::kData].get<xpu, 4, DType>(s);
118	    Shape<3> wmat_shape =
119	        Shape3(param_.num_group,
120	               param_.num_filter / param_.num_group,
121	               data.shape_[1] / param_.num_group * param_.kernel[0] * param_.kernel[1]);
122	    Tensor<xpu, 3, DType> wmat =
123	        in_data[conv::kWeight].get_with_shape<xpu, 3, DType>(wmat_shape, s);
124	    Tensor<xpu, 4, DType> out = out_data[conv::kOut].get<xpu, 4, DType>(s);
125	#if defined(__CUDACC__)
```
