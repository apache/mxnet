
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


