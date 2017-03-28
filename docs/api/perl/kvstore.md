# KVStore API

Topics:
* [Basic Push and Pull](#basic-push-and-pull)
* [List Key-Value Pairs](#list-key-value-pairs)

## Basic Push and Pull

Provides basic operation over multiple devices (GPUs) on a single device.

### Initialization

Let's consider a simple example. It initializes
a (int, NDArray) pair into the store, and then pulls the value out.

```perl
pdl> $kv = mx->kv->create('local')
pdl> $shape = [2,3]
pdl> $kv->init(3, mx->nd->ones($shape)*2)
pdl> $a = mx->nd->zeros($shape)
pdl> $kv->pull(3, out => $a)
pdl> print $a->aspdl
[
 [2 2 2]
 [2 2 2]
]
```

### Push, Aggregation, and Updater

For any key that's been initialized, you can push a new value with the same shape to the key, as follows:

```perl
pdl> $kv->push(3, mx->nd->ones($shape)*8)
pdl> $a = mx->nd->zeros($shape)
pdl> $kv->pull(3, out => $a)
pdl> print $a->aspdl
[
 [8 8 8]
 [8 8 8]
]
```

The data that you want to push can be stored on any device. Furthermore, you can push multiple
values into the same key, where KVStore first sums all of these
values, and then you pull the aggregated value, as follows:

```perl
pdl> $kv->push(3, [mx->nd->ones($shape, ctx=>mx->cpu(0)), mx->nd->ones($shape, ctx=>mx->cpu(1))])
pdl> $kv->pull(3, out => $a)
pdl> print $a->aspdl
[
 [2 2 2]
 [2 2 2]
]
```

For each push command, KVStore applies the pushed value to the value stored by an
`updater`. The default updater is `ASSIGN`. You can replace the default to
control how data is merged.

```perl
pdl> $updater = sub { my ($key, $input, $stored) = @_; print "update on key: $key\n"; $stored += $input * 3; }
pdl> $kv->_set_updater($updater)
pdl> $kv->push(3, [mx->nd->ones($shape, ctx=>mx->cpu(0)), mx->nd->ones($shape, ctx=>mx->cpu(1))])
update on key: 3
pdl> $kv->pull(3, out => $a)
pdl> print $a->aspdl
[
 [8 8 8]
 [8 8 8]
]
```

### Pull

You've already seen how to pull a single key-value pair. Similar to the way that you use the push command, you can
pull the value into several devices with a single call.

```perl
pdl> $b = [mx->nd->zeros($shape, ctx=>mx->cpu(0)), mx->nd->zeros($shape, ctx=>mx->cpu(1))]
pdl> $kv->pull(3, out => $b)
pdl> print $b->[1]->aspdl
[
 [8 8 8]
 [8 8 8]
]
```

## List Key-Value Pairs

All of the operations that we've discussed so far are performed on a single key. KVStore also provides
the interface for generating a list of key-value pairs. For a single device, use the following:

```perl
pdl> $keys = [5,7,9]
pdl> $kv->init($keys, [map { mx->nd->ones($shape) } 0..@$keys-1])
pdl> $kv->push($keys, [map { mx->nd->ones($shape) } 0..@$keys-1])
update on key: 5
update on key: 7
update on key: 9
pdl> $b = [map { mx->nd->ones($shape) } 0..@$keys-1]
pdl> $kv->pull($keys, out => $b)
pdl> print $b->[1]->aspdl
[
 [4 4 4]
 [4 4 4]
]
```
