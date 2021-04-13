<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# The NP on MXNet cheat sheet

To begin, import the `np` and `npx` module and update MXNet to run in
NumPy-like mode.

```{.python .input}
from mxnet import np, npx
npx.set_np()  # Change MXNet to the numpy-like mode.
```

NDArray figure (TODO)

## Creating arrays

```{.python .input}
np.array([1, 2, 3])  # default datatype is float32
```

```{.python .input}
np.array([(1.5, 2, 3), (4, 5, 6)], dtype='float16')
```

```{.python .input}
np.array([[(15,2,3), (4,5,6)], [(3,2,1), (4,5,6)]], dtype='int32')
```

### Initial placeholders

```{.python .input}
np.zeros((3, 4))  # Create an array of zeros
```

```{.python .input}
np.ones((2, 3, 4), dtype='int8')  # Create an array of ones
```

```{.python .input}
np.arange(10, 25, 5)  # Create an array of evenly spaced values (step value)
```

```{.python .input}
# Create an array of evenly spaced values (number of samples)
# np.linspace(0, 2, 9)
```

```{.python .input}
# np.full((2, 2), 7)  # Create a constant array
```

```{.python .input}
# np.eye(2)  # Create a 2X2 identity matrix
```

```{.python .input}
# np.random.random((2, 2))  # Create an array with random values
```

```{.python .input}
np.empty((3,2))  # Create an empty array
```

## I/O

### Saving and loading on disk

```{.python .input}
# Save one array
a = np.array([1, 2, 3])
npx.save('my_array', a)
npx.load('my_array')
```

```{.python .input}
# Save a list of arrays
b = np.array([4, 6, 8])
npx.save('my_arrays', [a, b])  # FIXME, cannot be a tuple
npx.load('my_arrays')
```

### Saving and loading text files

```{.python .input}
# np.loadtxt("myfile.txt")
# np.genfromtxt("my_file.csv", delimiter=',')
# np.savetxt("myarray.txt", a, delimiter=" ")
```

## Data types

```{.python .input}
# np.int64    # Signed 64-bit integer types
# np.float32  # Standard double-precision floating point
# np.complex  # Complex numbers represented by 128 floats
# np.bool     # Boolean type storing TRUE and FALSE values
# np.object   # Python object type
# np.string_  # Fixed-length string type
# np.unicode_ # Fixed-length unicode type
```

## Inspecting your array

```{.python .input}
a.shape # Array dimensions
```

```{.python .input}
len(a) # Length of array
```

```{.python .input}
b.ndim # Number of array dimensions
```

```{.python .input}
b.size # Number of array elements
```

```{.python .input}
b.dtype # Data type of array elements
```

```{.python .input}
# b.dtype.name # Name of data type
```

```{.python .input}
b.astype('int') # Convert an array to a different type
```

## Asking For Help

```{.python .input}
# np.info(np.ndarray.dtype)
```

## Array mathematics

### Arithmetic operations

```{.python .input}
a - b # Subtraction
```

```{.python .input}
np.subtract(a, b) # Subtraction
```

```{.python .input}
b + a # Addition
```

```{.python .input}
np.add(b, a) # Addition
```

```{.python .input}
a / b # Division
```

```{.python .input}
np.divide(a,b) # Division
```

```{.python .input}
a * b # Multiplication
```

```{.python .input}
np.multiply(a, b) # Multiplication
```

```{.python .input}
np.exp(b) # Exponentiation
```

```{.python .input}
np.sqrt(b) # Square root
```

```{.python .input}
np.sin(a) # Sines of an array
```

```{.python .input}
np.cos(b) # Element-wise cosine
```

```{.python .input}
np.log(a) # Element-wise natural logarithm
```

```{.python .input}
a.dot(b) # Dot product
```

### Comparison

### Aggregate functions

```{.python .input}
a.sum() # Array-wise sum
```

```{.python .input}
# a.min() # Array-wise minimum value
```

```{.python .input}
c = np.array(([[1,2,3], [2,3,4]]))
# c.max(axis=0) # Maximum value of an array row
```

```{.python .input}
# c.cumsum(axis=1) # Cumulative sum of the elements
```

```{.python .input}
a.mean() # Mean
```

```{.python .input}
# b.median() # Median
```

```{.python .input}
# a.corrcoef() # Correlation coefficient
```

```{.python .input}
# np.std(b) # Standard deviation
```

## Copying arrays

```{.python .input}
# a.view() # Create a view of the array with the same data
```

```{.python .input}
np.copy(a) # Create a copy of the array
```

```{.python .input}
a.copy() # Create a deep copy of the array
```

## Sorting Arrays

```{.python .input}
# a.sort() # Sort an array
```

```{.python .input}
# c.sort(axis=0) # Sort the elements of an array's axis
```

## Subsetting, slicing, indexing

### Subsetting

```{.python .input}
a[2] # Select the element at the 2nd index 3
```

```{.python .input}
c[0,1] # Select the element at row 1 column 2
```

### Slicing

```{.python .input}
a[0:2] # Select items at index 0 and 1
```

```{.python .input}
c[0:2,1] # Select items at rows 0 and 1 in column 1
```

```{.python .input}
c[:1] # Select all items at row 0
```

```{.python .input}
# c[1,...] # Same as [1,:,:]
```

```{.python .input}
a[ : :-1] #Reversed array a array([3, 2, 1])
```

### Boolean Indexing

```{.python .input}
# a[a<2] # Select elements from a less than 2
```

### Fancy indexing

```{.python .input}
c[[1,0,1,0], [0,1,2,0]] # Select elements (1,0),(0,1),(1,2) and (0,0)
```

```{.python .input}
c[[1,0,1,0]][:,[0,1,2,0]] # Select a subset of the matrix’s rows
```

## Array manipulation

### Transposing array

```{.python .input}
np.transpose(c) # Permute array dimensions
```

```{.python .input}
c.T # Permute array dimensions
```

### Changing array shape

```{.python .input}
# b.ravel() # Flatten the array
```

```{.python .input}
# c.reshape(3,-2) # Reshape, but don’t change data
```

### Adding and removing elements

```{.python .input}
# c.resize((6,2)) # Return a new array with shape (6, 2)
```

```{.python .input}
# np.append(h,g) # Append items to an array
```

```{.python .input}
# np.insert(a, 1, 5) # Insert items in an array
```

```{.python .input}
# np.delete(a, [1]) # Delete items from an array
```

### Combining arrays

```{.python .input}
np.concatenate((a,b),axis=0) # Concatenate arrays
```

```{.python .input}
# np.vstack((a,b)) # Stack arrays vertically (row-wise)
```

```{.python .input}
# np.r_[e,f] # Stack arrays vertically (row-wise)
```

```{.python .input}
# np.hstack((e,f)) # Stack arrays horizontally (column-wise)
```

```{.python .input}
# np.column_stack((a,d)) # Create stacked column-wise arrays
```

```{.python .input}
# np.c_[a,d] # Create stacked column-wise arrays
```

### Splitting arrays

```{.python .input}
# np.hsplit(a,3) # Split the array horizontally at the 3rd index
```

```{.python .input}
# np.vsplit(c,2) # Split the array vertically at the 2nd index
```

## Use GPUs

Prerequisites: A GPU exists and GPU-enabled MXNet is installed.

```{.python .input}
npx.num_gpus()  # Query number of GPUs
```

```{.python .input}
npx.gpu(0), npx.gpu(1)  # Context for the first and second GPUs
```

```{.python .input}
gpu_0 = npx.gpu(0) if npx.num_gpus() > 1 else npx.cpu()
g0 = np.zeros((2,3), ctx=gpu_0)  # Create array on GPU 0
g0
```

```{.python .input}
gpu_1 = npx.gpu(1) if npx.num_gpus() > 2 else npx.cpu()
g1 = np.random.uniform(size=(2,3), ctx=gpu_1)  # Create array on GPU 1
g1
```

```{.python .input}
# Copy to another GPU
g1.copyto(gpu_0)
```

```{.python .input}
# Return itself if matching the context, otherwise copy
g1.copyto(gpu_0), g1.copyto(gpu_0)
```

```{.python .input}
g1.context  # Query the device an array is on
```

```{.python .input}
## The computation is performed by the devices on which the input arrays are
g0 + g1.copyto(gpu_0)
```

## Auto differentiation

```{.python .input}
a.attach_grad() # Allocate gradient for a variable
a.grad # access the gradient
```

Compute the $\nabla_a b=\exp(2a)^T a$

```{.python .input}
from mxnet import autograd

with autograd.record():
    b = np.exp(2*a).dot(a)
b.backward()
a.grad
```

**Acknowledgement**

Adapted from www.datacamp.com.
