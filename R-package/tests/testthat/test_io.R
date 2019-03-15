# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

require(mxnet)

context("io")

source("get_data.R")

test_that("MNISTIter", {
  GetMNIST_ubyte()
  batch.size <- 100
  train_dataiter <- mx.io.MNISTIter(image = "data/train-images-idx3-ubyte", label = "data/train-labels-idx1-ubyte", 
    data.shape = c(784), batch.size = batch.size, shuffle = TRUE, flat = TRUE, 
    silent = 0, seed = 10)
  train_dataiter$reset()
  batch_count <- 0
  while (train_dataiter$iter.next()) {
    batch_count <- batch_count + 1
  }
  nbatch <- 60000/batch.size
  expect_equal(batch_count, nbatch)
  train_dataiter$reset()
  train_dataiter$iter.next()
  label_0 <- as.array(train_dataiter$value()$label)
  train_dataiter$iter.next()
  train_dataiter$iter.next()
  train_dataiter$iter.next()
  train_dataiter$iter.next()
  train_dataiter$reset()
  train_dataiter$iter.next()
  label_1 <- as.array(train_dataiter$value()$label)
  expect_equal(label_0, label_1)
})

test_that("Cifar10Rec", {
  GetCifar10()
  dataiter <- mx.io.ImageRecordIter(path.imgrec = "./data/cifar/train.rec", path.imglist = "./data/cifar/train.lst", 
    mean.img = "./data/cifar/cifar10_mean.bin", batch.size = 100, data.shape = c(28, 
      28, 3), rand.crop = TRUE, rand.mirror = TRUE)
  labelcount <- rep(0, 10)
  dataiter$reset()
  while (dataiter$iter.next()) {
    label <- as.array(dataiter$value()$label)
    for (i in label) {
      labelcount[i + 1] <- labelcount[i + 1] + 1
    }
  }
  
  expect_equal(labelcount, rep(5000, 10))
})

test_that("mx.io.arrayiter", {
  X <- matrix(c(1:10000), 100, 100)
  y <- c(1:100)
  dataiter <- mx.io.arrayiter(X, y, batch.size = 20, shuffle = FALSE)
  dataiter$reset()
  batch_count <- 0
  while (dataiter$iter.next()) {
    batch_count <- batch_count + 1
  }
  expect_equal(batch_count, 100/20)
  
  y <- round(y/10)
  dataiter <- mx.io.arrayiter(X, y, batch.size = 30, shuffle = FALSE)
  labelcount <- rep(0, 11)
  dataiter$reset()
  while (dataiter$iter.next()) {
    label <- as.array(dataiter$value()$label)
    for (i in label) {
      labelcount[i + 1] <- labelcount[i + 1] + 1
    }
  }
  
  expect_equal(labelcount, c(5, 9, 11, 9, 11, 9, 11, 13, 22, 14, 6))
})
