# RNN Example for MXNet Scala [2018/07]

This model has not been updated since 2017/07, the RNN example for LstmBucketing is no longer runnable. 
Please contribute to RNN support on Scala for the current version. If you would like to run this example for now, please revert your MXNet version to `v0.12`. 

- [ ] LstmBucketing
- [ ] TestCharRnn
- [x] TrainCharRNN

Please contact @lanking520 or @nswamy if you would like to dig in and fix this problem. The following is a way to reproduce the issues I found:

In summary, I suspect that some operator changes caused this issue.

## LSTM Bucketing
### Setup
Download the required ptb file from here:
```Bash
wget https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/ptb/ptb.train.txt
wget https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/ptb/ptb.valid.txt
```
You can use this [script](https://github.com/apache/incubator-mxnet/blob/master/scala-package/examples/scripts/rnn/run_lstm_bucketing.sh) to run the model 
or manually identify the file path by following [this](https://github.com/apache/incubator-mxnet/blob/master/scala-package/examples/src/main/scala/org/apache/mxnetexamples/rnn/LstmBucketing.scala#L36-L48)

#### Additional step to get it work
Please add the follows to [IO.scala](https://github.com/apache/incubator-mxnet/blob/master/scala-package/core/src/main/scala/org/apache/mxnet/IO.scala#L358)
```Scala
if (shapes.toIndexedSeq(0)._2.length == 2) {
   shapes.map { case (k, s) => new DataDesc(k, s, layout = "NT") }.toIndexedSeq
}
else shapes.map { case (k, s) => new DataDesc(k, s) }.toIndexedSeq
```

### Problem that I found
#### Segmentation fault caused by C++ backend
```Bash
#
# A fatal error has been detected by the Java Runtime Environment:
#
#  SIGSEGV (0xb) at pc=0x000000012d026fd8, pid=56081, tid=0x0000000000000e03
#
# JRE version: Java(TM) SE Runtime Environment (8.0_171-b11) (build 1.8.0_171-b11)
# Java VM: Java HotSpot(TM) 64-Bit Server VM (25.171-b11 mixed mode bsd-amd64 compressed oops)
# Problematic frame:
# C  [mxnet-scala+0x120afd8]
[error occurred during error reporting (printing problematic frame), id 0xe0000000]

# Failed to write core dump. Core dumps have been disabled. To enable core dumping, try "ulimit -c unlimited" before starting Java again
#
# An error report file with more information is saved as:
# /Users/qingla/Downloads/ITJ-Proj/hs_err_pid56081.log
Compiled method (nm)    2499 1793     n 0       org.apache.mxnet.LibInfo::mxNDArrayGetShape (native)
 total in heap  [0x00000001133e7c50,0x00000001133e7fc0] = 880
 relocation     [0x00000001133e7d78,0x00000001133e7db8] = 64
 main code      [0x00000001133e7dc0,0x00000001133e7fb8] = 504
 oops           [0x00000001133e7fb8,0x00000001133e7fc0] = 8
#
# If you would like to submit a bug report, please visit:
#   http://bugreport.java.com/bugreport/crash.jsp
# The crash happened outside the Java Virtual Machine in native code.
# See problematic frame for where to report the bug.
#

Process finished with exit code 134 (interrupted by signal 6: SIGABRT)

```
#### NDArray size mismatch
```Bash
2018-07-05 14:20:37,771 [main] [org.apache.mxnet.examples.rnn.LstmBucketing] [ERROR] - requirement failed: array size (320) do not match the size of NDArray (1)
java.lang.IllegalArgumentException: requirement failed: array size (320) do not match the size of NDArray (1)
	at scala.Predef$.require(Predef.scala:224)
	at org.apache.mxnet.NDArray.syncCopyfrom(NDArray.scala:623)
	at org.apache.mxnet.NDArray.set(NDArray.scala:755)
	at org.apache.mxnet.examples.rnn.BucketIo$BucketSentenceIter.next(BucketIo.scala:185)
	at org.apache.mxnet.module.BaseModule$$anonfun$fit$1.apply$mcVI$sp(BaseModule.scala:417)
	at scala.collection.immutable.Range.foreach$mVc$sp(Range.scala:160)
	at org.apache.mxnet.module.BaseModule.fit(BaseModule.scala:411)
	at org.apache.mxnet.examples.rnn.LstmBucketing$.main(LstmBucketing.scala:118)
	at org.apache.mxnet.examples.rnn.LstmBucketing.main(LstmBucketing.scala)
```
#### Data name not the same
```Bash
2018-07-05 14:41:46,508 [main] [org.apache.mxnet.examples.rnn.LstmBucketing] [ERROR] - Data provided by data_shapes don't match names specified by data_names (DataDesc[data,(32,20),float32,NT], DataDesc[l0_init_c_beta,(32,200),float32,NT], DataDesc[l1_init_c_beta,(32,200),float32,NT], DataDesc[l0_init_h_beta,(32,200),float32,NT], DataDesc[l1_init_h_beta,(32,200),float32,NT] vs. data)
java.lang.IllegalArgumentException: Data provided by data_shapes don't match names specified by data_names (DataDesc[data,(32,20),float32,NT], DataDesc[l0_init_c_beta,(32,200),float32,NT], DataDesc[l1_init_c_beta,(32,200),float32,NT], DataDesc[l0_init_h_beta,(32,200),float32,NT], DataDesc[l1_init_h_beta,(32,200),float32,NT] vs. data)
	at org.apache.mxnet.module.Module._checkNamesMatch(Module.scala:317)
	at org.apache.mxnet.module.Module._parseDataDesc(Module.scala:329)
	at org.apache.mxnet.module.Module.reshape(Module.scala:342)
	at org.apache.mxnet.module.Module.forward(Module.scala:450)
	at org.apache.mxnet.module.BucketingModule.forward(BucketingModule.scala:314)
	at org.apache.mxnet.module.BaseModule.forwardBackward(BaseModule.scala:153)
	at org.apache.mxnet.module.BaseModule$$anonfun$fit$1.apply$mcVI$sp(BaseModule.scala:420)
	at scala.collection.immutable.Range.foreach$mVc$sp(Range.scala:160)
	at org.apache.mxnet.module.BaseModule.fit(BaseModule.scala:411)
	at org.apache.mxnet.examples.rnn.LstmBucketing$.main(LstmBucketing.scala:118)
	at org.apache.mxnet.examples.rnn.LstmBucketing.main(LstmBucketing.scala)
2018-07-05 14:41:46,516 [Thread-0] [org.apache.mxnet.util.NativeLibraryLoader] [INFO] - Deleting /var/folders/qy/41_sjvss273_xcdjd_y74rzmzrh5b6/T/mxnet7527817581344011515/mxnet-scala
 --cpus VAL            : the cpus will be used, e.g. '0,1,2,3'
2018-07-05 14:41:46,516 [Thread-0] [org.apache.mxnet.util.NativeLibraryLoader] [INFO] - Deleting /var/folders/qy/41_sjvss273_xcdjd_y74rzmzrh5b6/T/mxnet7527817581344011515
```
## TestCharRNN
Please follow [this tutorial](https://mxnet.incubator.apache.org/tutorials/scala/char_lstm.html)

The problem I have

```Bash
2018-07-05 15:49:03,982 [main] [org.apache.mxnet.examples.rnn.TrainCharRnn] [ERROR] - [15:49:03] src/operator/contrib/../tensor/../elemwise_op_common.h:123: Check failed: assign(&dattr, (*vec)[i]) Incompatible attr in node  at 0-th output: expected [84,256], got [83,256]

Stack trace returned 10 entries:
[bt] (0) 0   mxnet-scala                         0x00000001287b0420 mxnet-scala + 25632
[bt] (1) 1   mxnet-scala                         0x00000001287b01cf mxnet-scala + 25039
[bt] (2) 2   mxnet-scala                         0x00000001288c2093 mxnet-scala + 1147027
[bt] (3) 3   mxnet-scala                         0x00000001288c1d16 mxnet-scala + 1146134
[bt] (4) 4   mxnet-scala                         0x00000001288c023b mxnet-scala + 1139259
[bt] (5) 5   mxnet-scala                         0x0000000129a83a41 mxnet-scala + 19765825
[bt] (6) 6   mxnet-scala                         0x0000000129a82439 mxnet-scala + 19760185
[bt] (7) 7   mxnet-scala                         0x00000001299d2745 mxnet-scala + 19040069
[bt] (8) 8   mxnet-scala                         0x00000001299d3707 mxnet-scala + 19044103
[bt] (9) 9   mxnet-scala                         0x000000012a01ae68 Java_org_apache_mxnet_LibInfo_mxImperativeInvoke + 424


org.apache.mxnet.MXNetError: [15:49:03] src/operator/contrib/../tensor/../elemwise_op_common.h:123: Check failed: assign(&dattr, (*vec)[i]) Incompatible attr in node  at 0-th output: expected [84,256], got [83,256]

Stack trace returned 10 entries:
[bt] (0) 0   mxnet-scala                         0x00000001287b0420 mxnet-scala + 25632
[bt] (1) 1   mxnet-scala                         0x00000001287b01cf mxnet-scala + 25039
[bt] (2) 2   mxnet-scala                         0x00000001288c2093 mxnet-scala + 1147027
[bt] (3) 3   mxnet-scala                         0x00000001288c1d16 mxnet-scala + 1146134
[bt] (4) 4   mxnet-scala                         0x00000001288c023b mxnet-scala + 1139259
[bt] (5) 5   mxnet-scala                         0x0000000129a83a41 mxnet-scala + 19765825
[bt] (6) 6   mxnet-scala                         0x0000000129a82439 mxnet-scala + 19760185
[bt] (7) 7   mxnet-scala                         0x00000001299d2745 mxnet-scala + 19040069
[bt] (8) 8   mxnet-scala                         0x00000001299d3707 mxnet-scala + 19044103
[bt] (9) 9   mxnet-scala                         0x000000012a01ae68 Java_org_apache_mxnet_LibInfo_mxImperativeInvoke + 424


	at org.apache.mxnet.Base$.checkCall(Base.scala:131)
	at org.apache.mxnet.NDArray$.genericNDArrayFunctionInvoke(NDArray.scala:101)
	at org.apache.mxnet.NDArray.copyTo(NDArray.scala:968)
	at org.apache.mxnet.examples.rnn.RnnModel$LSTMInferenceModel$$anonfun$3.apply(RnnModel.scala:44)
	at org.apache.mxnet.examples.rnn.RnnModel$LSTMInferenceModel$$anonfun$3.apply(RnnModel.scala:42)
	at scala.collection.Iterator$class.foreach(Iterator.scala:893)
	at scala.collection.AbstractIterator.foreach(Iterator.scala:1336)
	at scala.collection.MapLike$DefaultKeySet.foreach(MapLike.scala:174)
	at org.apache.mxnet.examples.rnn.RnnModel$LSTMInferenceModel.<init>(RnnModel.scala:42)
	at org.apache.mxnet.examples.rnn.TestCharRnn$.main(TestCharRnn.scala:62)
	at org.apache.mxnet.examples.rnn.TestCharRnn.main(TestCharRnn.scala)
2018-07-05 15:49:03,988 [Thread-0] [org.apache.mxnet.util.NativeLibraryLoader] [INFO] - Deleting /var/folders/qy/41_sjvss273_xcdjd_y74rzmzrh5b6/T/mxnet607484524537890023/mxnet-scala
2018-07-05 15:49:03,989 [Thread-0] [org.apache.mxnet.util.NativeLibraryLoader] [INFO] - Deleting /var/folders/qy/41_sjvss273_xcdjd_y74rzmzrh5b6/T/mxnet607484524537890023
```