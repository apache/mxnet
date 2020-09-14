# Step 3: Automatic differentiation with autograd

In this step, you learn how to use the MXNet `autograd` package to perform
gradient calculations by automatically calculating derivatives.

This is helpful because it will help you save time and effort. You train models
to get better as a function of experience. Usually, getting better means
minimizing a loss function. To achieve this goal, you often iteratively compute
the gradient of the loss with respect to weights and then update the weights
accordingly. Gradient calculations are straightforward through a chain rule.
However, for complex models, working this out manually is challenging.

The `autograd` package helps you by automatically calculating derivatives.

## Basic use

To get started, import the `autograd` package as in the following code.

```{.python .input  n=1}
from mxnet import np, npx
from mxnet import autograd
npx.set_np()
```

As an example, you could differentiate a function $f(x) = 2 x^2$ with respect to
parameter $x$. You can start by assigning an initial value of $x$, as follows:

```{.python .input  n=2}
x = np.array([[1, 2], [3, 4]])
x
```

```{.json .output n=2}
[
 {
  "data": {
   "text/plain": "array([[1., 2.],\n       [3., 4.]])"
  },
  "execution_count": 2,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

After you compute the gradient of $f(x)$ with respect to $x$, you need a place
to store it. In MXNet, you can tell an ndarray that you plan to store a gradient
by invoking its `attach_grad` method, shown in the following example.

```{.python .input  n=3}
x.attach_grad()
```

Next, define the function $y=f(x)$. To let MXNet store $y$, so that you can
compute gradients later, use the following code to put the definition inside an
`autograd.record()` scope.

```{.python .input  n=4}
with autograd.record():
    y = 2 * x * x
```

You can invoke back propagation (backprop) by calling `y.backward()`. When $y$
has more than one entry, `y.backward()` is equivalent to `y.sum().backward()`.

```{.python .input  n=5}
y.backward()
```

Next, verify whether this is the expected output. Note that $y=2x^2$ and
$\frac{dy}{dx} = 4x$, which should be `[[4, 8],[12, 16]]`. Check the
automatically computed results.

```{.python .input  n=6}
x.grad
```

```{.json .output n=6}
[
 {
  "data": {
   "text/plain": "array([[ 4.,  8.],\n       [12., 16.]])"
  },
  "execution_count": 6,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

Now let's dive into `y.backward()` by first discussing a bit on gradients.
First, as we alluded to earlier `y.backward()` is equivalent to
`y.sum().backward()`.

```{.python .input  n=7}
with autograd.record():
    y = np.sum(2 * x * x)
y.backward()
x.grad
```

```{.json .output n=7}
[
 {
  "data": {
   "text/plain": "array([[ 4.,  8.],\n       [12., 16.]])"
  },
  "execution_count": 7,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

In order to briefly understand this it is helpful to know what our ndarrays
essentially are. Basically, our ndarrays are classes with a forward and backward
method. Where we the number of args in `backward()` must equal the number of
items returned in forward. Additionally, we can customize backward if we so
desire.

## Custom MXNet ndarray operations

In order to understand the `backward()` method it is beneficial to first
understand how can create custom operations.

```{.python .input  n=8}
class My_First_Custom_Operation(autograd.Function):
    def __init__(self):
        super().__init__()
    def forward(self,x,y):
        self.save_for_backward(x,y) #Save previously calculated values
        return 2*x,2*x*y,2*y
    def backward(self,dx,dxy,dy):
        """
        The input number of arguments must match the number of outputs from forward.
        Furthermore, the number of output arguments must match the number of inputs from forward.
        """
        x,y = self.saved_tensors #Use previously calculated values
        return x,y
```

Now that we have created our first

```{.python .input  n=10}
with autograd.record():
    x = np.random.uniform(-1,1,(2,3))
    y = np.random.uniform(-1,1,(2,3))
    x.attach_grad()
    y.attach_grad()
    z = My_First_Custom_Operation()
    z1,z2,z3 = z(x,y)
    out = z1 + z2 + z3
out.backward()
print(np.array_equiv(x.asnumpy(),x.asnumpy()))
print(np.array_equiv(y.asnumpy(),y.asnumpy()))
```

```{.json .output n=10}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "True\nTrue\n"
 }
]
```

Alternatively, we could wrap special effects in a normal function like this.

```{.python .input  n=11}
def my_first_function(x):
    if autograd.is_recording() & (not autograd.is_training()): # Return if we are recording
        return(2*x)
    elif autograd.is_training(): # Return something else when training
        return(4*x)
    else:
        return(x)
```

```{.python .input  n=12}
y = my_first_function(x)
print(np.array_equiv(y.asnumpy(),x.asnumpy()))
with autograd.record(train_mode=False):
    y = my_first_function(x)
y.backward()
print(x.grad)
with autograd.record(train_mode=True):# train_mode = True by default
    y = my_first_function(x)
y.backward()
print(x.grad)
```

```{.json .output n=12}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "True\n[[2. 2. 2.]\n [2. 2. 2.]]\n[[4. 4. 4.]\n [4. 4. 4.]]\n"
 }
]
```

We could create functions with `autograd.record()`.

```{.python .input  n=13}
def my_second_function(x):
    with autograd.record():
        return(2*x)
```

```{.python .input  n=14}
y = my_second_function(x)
y.backward()
print(x.grad)
```

```{.json .output n=14}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "[[2. 2. 2.]\n [2. 2. 2.]]\n"
 }
]
```

We can also combine multiple functions.

```{.python .input  n=15}
y = my_second_function(x)
with autograd.record():
    z = my_second_function(y) + 2
z.backward()
print(x.grad)
```

```{.json .output n=15}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "[[4. 4. 4.]\n [4. 4. 4.]]\n"
 }
]
```

Additionally, MXNet records the execution trace and computes the gradient
accordingly.Consider the following function `f` in the following example code.
The function doubles the inputs until its `norm` reaches 1000. Then it selects
one element depending on the sum of its elements.

```{.python .input  n=24}
def f(a):
    b = a * 2
    while np.abs(b).sum() < 1000:
        b = b * 2
    if b.sum() >= 0:
        c = b[0]
    else:
        c = b[1]
    return c
```

In this example, you record the trace and feed in a random value.

```{.python .input  n=25}
a = np.random.uniform(size=2)
a.attach_grad()
with autograd.record():
    c = f(a)
c.backward()
```

You can see that `b` is a linear function of `a`, and `c` is chosen from `b`.
The gradient with respect to `a` be will be either `[c/a[0], 0]` or `[0,
c/a[1]]`, depending on which element from `b` is picked. You see the results of
this example with this code:

```{.python .input  n=27}
a.grad == c/a
```

```{.json .output n=27}
[
 {
  "data": {
   "text/plain": "array([ True, False])"
  },
  "execution_count": 27,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## Understanding arguments of Autograd

An interesting fact is that `autograd` performs a `sum()` operation on vector
valued outputs. You can see that here.

```{.python .input  n=16}
with autograd.record():
    y = np.sum(2 * x * x)
    z = 2 * x * x
y.backward()
y = x.grad
z.backward()
np.array_equiv(x.grad.asnumpy(),y.asnumpy())
```

```{.json .output n=16}
[
 {
  "data": {
   "text/plain": "True"
  },
  "execution_count": 16,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

Although, this may seem unexpected at first the key is to understand what is
happening inside of `autograd`. To understand this you have to understand
Jacobians and multivariate differentiation. What is the derivative if we had a
vector valued output?

```{.python .input  n=17}
x = np.random.uniform(-1,1,(2,3))
y = np.random.uniform(-1,1,(2,3))
x.attach_grad()
y.attach_grad()

with autograd.record():
    z = 2*x,2*x*y,2*x+2*y
    z1,z2,z3 = z
z1.backward()
v = x.grad.copy()
v1 = y.grad.copy()
print(x.grad)
print(y.grad)
z2.backward()
v += x.grad
v1 += y.grad
print(x.grad)
print(y.grad)
z3.backward()
print(x.grad)
print(y.grad)
v += x.grad
v1 += y.grad
print()
print(v)
print(v1)
```

```{.json .output n=17}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "[[2. 2. 2.]\n [2. 2. 2.]]\n[[0. 0. 0.]\n [0. 0. 0.]]\n[[-0.46623397 -0.08933949  1.1669002 ]\n [ 1.2486749   0.11557961 -0.08009136]]\n[[-0.2496512  -0.80986154  1.567092  ]\n [-1.7731481   1.854651   -0.90937483]]\n[[2. 2. 2.]\n [2. 2. 2.]]\n[[2. 2. 2.]\n [2. 2. 2.]]\n\n[[3.533766  3.9106605 5.1669   ]\n [5.248675  4.1155796 3.9199085]]\n[[1.7503488  1.1901385  3.567092  ]\n [0.22685194 3.854651   1.0906252 ]]\n"
 }
]
```

As you can see one way we can deal with multiple different gradients is to `sum`
like we did with `v,v1`. This is what autograd does by default. As you can see
below.

```{.python .input  n=18}
with autograd.record():
    z = 2*x,2*x*y,2*x+2*y
    z = np.concatenate([np.expand_dims(zi,axis=0) for zi in z],axis=0)
print(z.shape)
z.backward()
print(np.array_equiv(x.grad.asnumpy(),v.asnumpy()))
print(np.array_equiv(y.grad.asnumpy(),v1.asnumpy()))
```

```{.json .output n=18}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "(3, 2, 3)\nTrue\nTrue\n"
 }
]
```

Now if we were to sum along the first dimension (which autograd assumes is the
batch_dimension).

```{.python .input  n=19}
with autograd.record():
    z = 2*x,2*x*y,2*x+2*y
    z = np.sum(np.concatenate([np.expand_dims(zi,axis=0) for zi in z],axis=0),axis=0)
print(z.shape)
z.backward()
print(np.array_equiv(x.grad.asnumpy(),v.asnumpy()))
print(np.array_equiv(y.grad.asnumpy(),v1.asnumpy()))
```

```{.json .output n=19}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "(2, 3)\nTrue\nTrue\n"
 }
]
```

So why do we automatically `sum`? Well as you can see we could have a problem
where our shapes wouldn't match. Essentially, if we do the operations all
seperately then we get 3 different gradients for `x,y`. Which one is correct?
Therefore, we have to decide how we should combine those operations. This is
where autograd plays a role by default just summing the arguments. We could just
perform `mean` if we wanted the average, like so.

```{.python .input  n=20}
with autograd.record():
    z = 2*x,2*x*y,2*x+2*y
    z = np.mean(np.concatenate([np.expand_dims(zi,axis=0) for zi in z],axis=0),axis=0)
print(z.shape)
z.backward()
print(np.array_equiv(x.grad.asnumpy(),v.asnumpy()))
print(np.array_equiv(y.grad.asnumpy(),v1.asnumpy()))
v = x.grad.copy()
v1 = y.grad.copy()
```

```{.json .output n=20}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "(2, 3)\nFalse\nFalse\n"
 }
]
```

Alternatively, we could pass in an array for the gradient with respect to our
current location "upstream". Please note, the shape must match the output shape.

```{.python .input  n=21}
with autograd.record():
    z = 2*x,2*x*y,2*x+2*y
    z = np.concatenate([np.expand_dims(zi,axis=0) for zi in z],axis=0)
print(z.shape)
z.backward(np.full(z.shape,1./3.))
print(np.array_equiv(x.grad.asnumpy(),v.asnumpy()))
print(np.array_equiv(y.grad.asnumpy(),v1.asnumpy()))
```

```{.json .output n=21}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "(3, 2, 3)\nTrue\nTrue\n"
 }
]
```

As you can see there are 3 values along the dimension 0, so taking a `mean`
along this access is the same as summing that axis and multiplying by `1/3`.

## Advanced MXNet ndarray operations with Autograd

Additionally, we can control gradients for different ndarray operations. For
instance, perhaps I want to check that the gradients are propogating properly?

```{.python .input  n=22}
with autograd.record():
    y = 3*x
    y=y.detach()
    y.attach_grad()
    z = 4*y+2*x
z.backward()
print(x.grad)
print(y.grad)
```

```{.json .output n=22}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "[[2. 2. 2.]\n [2. 2. 2.]]\n[[4. 4. 4.]\n [4. 4. 4.]]\n"
 }
]
```

Additionally, if you want to output multiple values through a custom operation
you need to retain the graph.

```{.python .input  n=23}
class My_First_Custom_Operation(autograd.Function):
    def __init__(self):
        super().__init__()
    def forward(self,x,y):
        self.save_for_backward(x,y)
        return 2*x,2*x*y,2*y
    def backward(self,dx,dxy,dy):
        x,y = self.saved_tensors
        return x,y
    
x = np.random.uniform(-1,1,(2,3))
y = np.random.uniform(-1,1,(2,3))
x.attach_grad()
y.attach_grad()
with autograd.record():
    z = My_First_Custom_Operation()
    z1,z2,z3 = z(x,y)
    z1 = z1 + z2
    z3 = z2 + z3
z1.backward(retain_graph=True)
print(x.grad)
print(y.grad)
z3.backward()
print(x.grad)
print(y.grad)
```

```{.json .output n=23}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "[[ 0.13608909 -0.21443039  0.8511933 ]\n [ 0.6721575  -0.8579279  -0.32520765]]\n[[-0.8257414   0.2963438  -0.9595632 ]\n [-0.2635169   0.6652397   0.91431034]]\n[[ 0.13608909 -0.21443039  0.8511933 ]\n [ 0.6721575  -0.8579279  -0.32520765]]\n[[-0.8257414   0.2963438  -0.9595632 ]\n [-0.2635169   0.6652397   0.91431034]]\n"
 }
]
```

## Next steps

Learn how to construct a neural network with the Gluon module: [Step 2: Create a
neural network](2-nn.md).
