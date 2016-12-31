# NCE Loss
This tutorial shows how to use nce-loss to speed up multi-class classification when the number of classes is huge.

You can get the source code for this example on [GitHub](https://github.com/dmlc/mxnet/tree/master/example/nce-loss).

## Toy Examples

* toy_softmax.py. A multi class example using softmax output
* toy_nce.py. A multi-class example using nce loss

### Word2Vec

* word2vec.py. A CBOW word2vec example using nce loss

Run word2vec.py with the following command:

```
    ./get_text8.sh
    python word2vec.py
```

### LSTM

* lstm_word.py. An LSTM example using nce loss

Run lstm_word.py with the  following command:

```
    ./get_text8.sh
    python lstm_word.py
```

## References

For more details, see [http://www.jianshu.com/p/e439b43ea464](http://www.jianshu.com/p/e439b43ea464) (in Chinese).

## Next Steps
* [MXNet tutorials index](http://mxnet.io/tutorials/index.html)