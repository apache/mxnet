##NDArray相当于Opencv 中的Mat类型
其定义如下
![Ndarray](.\pic\ndarray.png)
-	shape: 维数的大小一般可以用如下方式定义
			mshadow::TShape shape = mshadow::Shape2(3, 4)
其中shape2可以替换为shape1,shape3,shape4
-	ctx:表示数据存储或者运算的位置
			mxnet::Context ctx = mxnet::Context::Create(mxnet::Context::kCPU, 1);
其中kCPU可以换做kGPU和kCPUPinned(pinned CPU context.)，数字1代表cpu核1，从0开始计数
-	delay_alloc：是否延迟分配

####NDArrray的赋值和运算
整体赋值
```CPP
vector<float> data_vec;
const float *dptr = data_vec.data();
NDArray data_array = NDArray(mshadow::Shape4(1000, 1, 28, 28), ctx_cpu,
			false);
data_array.SyncCopyFromCPU(dptr, 1000 * 28* 28);
data_array.WaitToRead();
```
利用指针操作
```cpp
  mxnet::NDArray a(f, d, false);
  mxnet::NDArray b(f, d, false);
  mxnet::real_t* aptr = static_cast<mxnet::real_t*>(a.data().dptr_);
  mxnet::real_t* bptr = static_cast<mxnet::real_t*>(b.data().dptr_);
  for (int i = 0; i < m; i++)
  {
    for (int j = 0; j < n; j++)
    {
      aptr[i*n + j] = i*n + j;
      bptr[i*n + j] = i*n + j;
    }
  }
  mxnet::NDArray c = a + b;
  // this is important, wait for the execution to complete before reading
  c.WaitToRead();//等待之前所有操作完成再读取c的值
```
对于gpu模式，使用指针操作会造成错误，因此在进行指针操作时，需要定义一个与其相同的NDArray放置在CPU中，并使用

	mxnet::NDArray c_cpu =ctx_gpu.Copy(ctx_cpu);
NDarray 提供Reshape和Slice（提取该Ndarray的特定部分，开始和结束的位置均表示第一维的位置）
NDarray提供基本的加减乘除操作，均是点操作

