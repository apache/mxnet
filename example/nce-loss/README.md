#Examples of NCE Loss

nce-loss is used to speedup multi-class classification when class num is huge.

## Toy example

* toy_softmax.py: a multi class example using softmax output
* toy_nce.py: a multi-class example using nce loss

## Word2Vec

* word2vec.py: a CBOW word2vec example using nce loss

You can run it by

```
./get_text8.sh
python word2vec.py

```

## LSTM

* lstm_word.py: a lstm example use nce loss

You can run it by

```
./get_text8.sh
python lstm_word.py
```

## References

You can refer to [http://www.jianshu.com/p/e439b43ea464](http://www.jianshu.com/p/e439b43ea464) for more details. (In Chinese)
