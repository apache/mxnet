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


G_iterator<- function(batch_size){
  
  batch<- 0
  batch_per_epoch<-5
  
  reset<- function(){
    batch<<- 0
  }
  
  iter.next<- function(){
    batch<<- batch+1
    if (batch>batch_per_epoch) {
      return(FALSE)
    } else {
      return(TRUE)
    }
  }
  
  value<- function(){
    set.seed(123+batch)
    digit<- mx.nd.array(sample(0:9, size = batch_size, replace = T))
    data<- mx.nd.one.hot(indices = digit, depth = 10)
    data<- mx.nd.reshape(data = data, shape = c(1,1,-1, batch_size))
    return(list(data=data, digit=digit))
  }
  
  return(list(reset=reset, iter.next=iter.next, value=value, batch_size=batch_size, batch=batch))
}

D_iterator<- function(batch_size){
  
  batch<- 0
  batch_per_epoch<-5
  
  reset<- function(){
    batch<<- 0
  }
  
  iter.next<- function(){
    batch<<- batch+1
    if (batch>batch_per_epoch) {
      return(FALSE)
    } else {
      return(TRUE)
    }
  }
  
  value<- function(){
    set.seed(123+batch)
    idx<- sample(length(train_label), size = batch_size, replace = T)
    data<- train_data[,,,idx, drop=F]
    label<- mx.nd.array(train_label[idx])
    digit<- mx.nd.one.hot(indices = label, depth = 10)
    
    return(list(data=mx.nd.array(data), digit=digit, label=label))
  }
  
  return(list(reset=reset, iter.next=iter.next, value=value, batch_size=batch_size, batch=batch))
}


