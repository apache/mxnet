
# NDArray Indexing - Array indexing features

MXNet's advanced indexing features are modeled after [NumPy's implementation and documentation](https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.indexing.html#combining-advanced-and-basic-indexing). You will see direct adaptations of many NumPy indexing features and examples which are close, if not identical, so we borrow much from their documentation.

`NDArray`s can be indexed using the standard Python `x[obj]` syntax, where _x_ is the array and _obj_ the selection.

There are two kinds of indexing available:

1. basic slicing
1. advanced indexing

In MXNet, we support both basic and advanced indexing following the convention of indexing NumPy's `ndarray`.


## Basic Slicing and Indexing

Basic slicing extends Python’s basic concept of slicing to N dimensions. For a quick review:

```
a[start:end] # items start through end-1
a[start:]    # items start through the rest of the array
a[:end]      # items from the beginning through end-1
a[:]         # a copy of the whole array
```


```python
from mxnet import nd
```

For some working examples of basic slicing we'll start simple.


```python
x = nd.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='int32')
x[5:]
```





    [5 6 7 8 9]
    <NDArray 5 @cpu(0)>




```python
x = nd.array([0, 1, 2, 3])
print('1D complete array, x=', x)
s = x[1:3]
print('slicing the 2nd and 3rd elements, s=', s)
```

    1D complete array, x=
    [ 0.  1.  2.  3.]
    <NDArray 4 @cpu(0)>
    slicing the 2nd and 3rd elements, s=
    [ 1.  2.]
    <NDArray 2 @cpu(0)>


Now let's try slicing the 2nd and 3rd elements of a multi-dimensional array.


```python
x = nd.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print('multi-D complete array, x=', x)
s = x[1:3]
print('slicing the 2nd and 3rd elements, s=', s)
```

    multi-D complete array, x=
    [[  1.   2.   3.   4.]
     [  5.   6.   7.   8.]
     [  9.  10.  11.  12.]]
    <NDArray 3x4 @cpu(0)>
    slicing the 2nd and 3rd elements, s=
    [[  5.   6.   7.   8.]
     [  9.  10.  11.  12.]]
    <NDArray 2x4 @cpu(0)>


Now let's try writing to a specific element. We'll write `9` to element `2` using `x[2] = 9.0`, which will update the whole row.


```python
print('original x, x=', x)
x[2] = 9.0
print('replaced entire row with x[2] = 9.0, x=', x)
```

    original x, x=
    [[  1.   2.   3.   4.]
     [  5.   6.   7.   8.]
     [  9.  10.  11.  12.]]
    <NDArray 3x4 @cpu(0)>
    replaced entire row with x[2] = 9.0, x=
    [[ 1.  2.  3.  4.]
     [ 5.  6.  7.  8.]
     [ 9.  9.  9.  9.]]
    <NDArray 3x4 @cpu(0)>


We can target specific elements too. Let's replace the number `3` in the first row with the number `9` using `x[0, 2] = 9.0`.


```python
print('original x, x=', x)
x[0, 2] = 9.0
print('replaced specific element with x[0, 2] = 9.0, x=', x)
```

    original x, x=
    [[ 1.  2.  3.  4.]
     [ 5.  6.  7.  8.]
     [ 9.  9.  9.  9.]]
    <NDArray 3x4 @cpu(0)>
    replaced specific element with x[0, 2] = 9.0, x=
    [[ 1.  2.  9.  4.]
     [ 5.  6.  7.  8.]
     [ 9.  9.  9.  9.]]
    <NDArray 3x4 @cpu(0)>


Now lets target even more by selecting a couple of targets at the same time. We'll replace the `6` and the `7` with `x[1:2, 1:3] = 5.0`.


```python
print('original x, x=', x)
x[1:2, 1:3] = 5.0
print('replaced range of elements with x[1:2, 1:3] = 5.0, x=', x)
```

    original x, x=
    [[ 1.  2.  9.  4.]
     [ 5.  6.  7.  8.]
     [ 9.  9.  9.  9.]]
    <NDArray 3x4 @cpu(0)>
    replaced range of elements with x[1:2, 1:3] = 5.0, x=
    [[ 1.  2.  9.  4.]
     [ 5.  5.  5.  8.]
     [ 9.  9.  9.  9.]]
    <NDArray 3x4 @cpu(0)>


## New Indexing Features in v1.0

### Step

The basic slice syntax is `i:j:k` where _i_ is the starting index, _j_ is the stopping index, and _k_ is the step (_k_ must be nonzero).

**Note**: Previously, MXNet supported basic slicing and indexing only with `step=1`. From release 1.0, arbitrary values of `step` are supported.


```python
x = nd.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='int32')
# Select elements 1 through 7, and use a step of 2
x[1:7:2]
```





    [1 3 5]
    <NDArray 3 @cpu(0)>



## Negative Indices
Negative _i_ and _j_ are interpreted as _n + i_ and _n + j_ where _n_ is the number of elements in the corresponding dimension. Negative _k_ makes stepping go towards smaller indices.


```python
x[-2:10]
```





    [8 9]
    <NDArray 2 @cpu(0)>



If the number of objects in the selection tuple is less than N , then : is assumed for any subsequent dimensions.


```python
x = nd.array([[[1],[2],[3]],
              [[4],[5],[6]]], dtype='int32')
x[1:2]
```





    [[[4]
      [5]
      [6]]]
    <NDArray 1x3x1 @cpu(0)>



You may use slicing to set values in the array, but (unlike lists) you can never grow the array. The size of the value to be set in `x[obj] = value` must be able to broadcast to the same shape as `x[obj]`.


```python
x = nd.arange(16, dtype='int32').reshape((4, 4))
print(x)
```


    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]
     [12 13 14 15]]
    <NDArray 4x4 @cpu(0)>



```python
print(x[1:4:2, 3:0:-1])
```


    [[ 7  6  5]
     [15 14 13]]
    <NDArray 2x3 @cpu(0)>



```python
x[1:4:2, 3:0:-1] = [[16], [17]]
print(x)
```


    [[ 0  1  2  3]
     [ 4 16 16 16]
     [ 8  9 10 11]
     [12 17 17 17]]
    <NDArray 4x4 @cpu(0)>


## New Advanced Indexing Features in v1.0

Advanced indexing is triggered when the selection object, obj, is a non-tuple sequence object (e.g. a Python list), a NumPy `ndarray` (of data type integer), an MXNet `NDArray`, or a tuple with at least one sequence object.

Advanced indexing always returns a __copy__ of the data.

**Note**:
- When the selection object is a Python list, it must be a list of integers. MXNet does not support the selection object being a nested list. That is, `x[[1, 2]]` is supported, while `x[[1], [2]]` is not.
- When the selection object is a NumPy `ndarray` or an MXNet `NDArray`, there is no dimension restrictions on the object.
- When the selection object is a tuple containing Python list(s), both integer lists and nested lists are supported. That is, both `x[1:4, [1, 2]]` and `x[1:4, [[1], [2]]` are supported.

### Purely Integer Array Indexing
When the index consists of as many integer arrays as the array being indexed has dimensions, the indexing is straight forward, but different from slicing.

Advanced indexes always are [broadcast](https://docs.scipy.org/doc/numpy-1.13.0/reference/ufuncs.html#ufuncs-broadcasting) and iterated as one:
```python
result[i_1, ..., i_M] == x[ind_1[i_1, ..., i_M], ind_2[i_1, ..., i_M],
                           ..., ind_N[i_1, ..., i_M]]
```
Note that the result shape is identical to the (broadcast) indexing array shapes `ind_1, ..., ind_N`.

**Example:**
From each row, a specific element should be selected. The row index is just [0, 1, 2] and the column index specifies the element to choose for the corresponding row, here [0, 1, 0]. Using both together the task can be solved using advanced indexing:


```python
x = nd.array([[1, 2],
              [3, 4],
              [5, 6]], dtype='int32')
x[[0, 1, 2], [0, 1, 0]]
```





    [1 4 5]
    <NDArray 3 @cpu(0)>



To achieve a behavior similar to the basic slicing above, broadcasting can be used. This is best understood with an example.

Example:
From a 4x3 array the corner elements should be selected using advanced indexing. Thus all elements for which the column is one of `[0, 2]` and the row is one of `[0, 3]` need to be selected. To use advanced indexing one needs to select all elements explicitly. Using the method explained previously one could write:


```python
x = nd.array([[ 0,  1,  2],
              [ 3,  4,  5],
              [ 6,  7,  8],
              [ 9, 10, 11]], dtype='int32')
x[[[0, 0], [3, 3]],
  [[0, 2], [0, 2]]]
```





    [[ 0  2]
     [ 9 11]]
    <NDArray 2x2 @cpu(0)>



However, since the indexing arrays above just repeat themselves, broadcasting can be used.


```python
x[[[0], [3]],
  [[0, 2]]]
```





    [[ 0  2]
     [ 9 11]]
    <NDArray 2x2 @cpu(0)>



### Combining Advanced and Basic Indexing
There are three situations we need to consider when mix advanced and basic indices in a single selection object. Let's look at examples to understand each one's behavior.

- There is only one advanced index in the selection object. For example, `x` is an `NDArray` with `shape=(10, 20, 30, 40, 50)` and `result=x[:, :, ind]` has one advanced index `ind` with `shape=(2, 3, 4)` on the third axis. The `result` will have `shape=(10, 20, 2, 3, 4, 40, 50)` because the subspace of `x` in the third dimension is replaced by the subspace of `shape=(2, 3, 4)`. If we let _i_, _j_, _k_ loop over the (2, 3, 4)-shaped subspace, it is equivalent to `result[:, :, i, j, k, :, :] = x[:, :, ind[i, j, k], :, :]`.


```python
import numpy as np
shape = (10, 20, 30, 40, 50)
x = nd.arange(np.prod(shape), dtype='int32').reshape(shape)
ind = nd.arange(24).reshape((2, 3, 4))
print(x[:, :, ind].shape)
```

    (10, 20, 2, 3, 4, 40, 50)


- There are at least two advanced indices in the selection object, and all the advanced indices are adjacent to each other. For example, `x` is an `NDArray` with `shape=(10, 20, 30, 40, 50)` and `result=x[:, :, ind1, ind2, :]` has two advanced indices with shapes that are broadcastable to `shape=(2, 3, 4)`. Then the `result` has `shape=(10, 20, 2, 3, 4, 50)` because `(30, 40)`-shaped subspace has been replaced with `(2, 3, 4)`-shaped subspace from the indices.


```python
ind1 = [0, 1, 2, 3]
ind2 = [[[0], [1], [2]], [[3], [4], [5]]]
print(x[:, :, ind1, ind2, :].shape)
```

    (10, 20, 2, 3, 4, 50)


- There are at least two advanced indices in the selection object, and there is at least one advanced index separated from the others by basic indices. For example,  `x` is an `NDArray` with `shape=(10, 20, 30, 40, 50)` and `result=x[:, :, ind1, :, ind2]` has two advanced indices with shapes that are broadcastable to `shape=(2, 3, 4)`. Then the `result` has `shape=(2, 3, 4, 10, 20, 40)` because there is no unambiguous place to place the indexing subspace, hence it is prepended to the beginning.


```python
print(x[:, :, ind1, :, ind2].shape)
```

    (2, 3, 4, 10, 20, 40)

## References

[NumPy documentation](https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.indexing.html#combining-advanced-and-basic-indexing)

<!-- INSERT SOURCE DOWNLOAD BUTTONS -->
