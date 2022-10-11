# 0022_fastai_pt2_2019_why_sqrt5
---
skip_exec: true
---

```
%load_ext autoreload
%autoreload 2

%matplotlib inline
```

## Does nn.Conv2d init work well?
### [00:00](https://youtu.be/AcA8HAYh7IE?list=PLfYUBJiXbdtTIdtE1U8qgyxo4Jy2Y91uj&t=0) one of the purpose of part 2 is to demonstrate how Jeremy does research; Jeremy is going to show us how he does research to find out how well does a mysterious line of code in the pytorch work

[01:28](https://youtu.be/AcA8HAYh7IE?list=PLfYUBJiXbdtTIdtE1U8qgyxo4Jy2Y91uj&t=88) - 

[02:28](https://youtu.be/AcA8HAYh7IE?list=PLfYUBJiXbdtTIdtE1U8qgyxo4Jy2Y91uj&t=148) - how to resize a 3 color channel image into a single changel image 28x28

[03:06](https://youtu.be/AcA8HAYh7IE?list=PLfYUBJiXbdtTIdtE1U8qgyxo4Jy2Y91uj&t=186) - when would Jeremy create a function during research; experiment to show that the line is not performing well

[08:55](https://youtu.be/AcA8HAYh7IE?list=PLfYUBJiXbdtTIdtE1U8qgyxo4Jy2Y91uj&t=535) - Jeremy writing his own version of kaiming init

[15:59](https://youtu.be/AcA8HAYh7IE?list=PLfYUBJiXbdtTIdtE1U8qgyxo4Jy2Y91uj&t=959) - Jeremy reimplemented what pytorch had on kaiming init; Jeremy used an example to test on how useless or useful of the line in pytorch;

[17:30](https://youtu.be/AcA8HAYh7IE?list=PLfYUBJiXbdtTIdtE1U8qgyxo4Jy2Y91uj&t=1050) -  using kaiming_uniform_ to test the line, and the result is better but still problematic

[18:58](https://youtu.be/AcA8HAYh7IE?list=PLfYUBJiXbdtTIdtE1U8qgyxo4Jy2Y91uj&t=1138) - look at 2b why need a good init; why in the past neuralnet is so hard to train; why weights initialization is so crucial to training or learning

[21:04](https://youtu.be/AcA8HAYh7IE?list=PLfYUBJiXbdtTIdtE1U8qgyxo4Jy2Y91uj&t=1264) - Sylvian further explained something interesting 

[21:30](https://youtu.be/AcA8HAYh7IE?list=PLfYUBJiXbdtTIdtE1U8qgyxo4Jy2Y91uj&t=1290) - how pytorch team responded

[23:52](https://youtu.be/AcA8HAYh7IE?list=PLfYUBJiXbdtTIdtE1U8qgyxo4Jy2Y91uj&t=1433) - many init papers and approaches

[27:11](https://youtu.be/AcA8HAYh7IE?list=PLfYUBJiXbdtTIdtE1U8qgyxo4Jy2Y91uj&t=1631) - ground up so that we can ask questions on pytorch strange and historical edges

[28:56](https://youtu.be/AcA8HAYh7IE?list=PLfYUBJiXbdtTIdtE1U8qgyxo4Jy2Y91uj&t=1736) - let's train a model with our fully connected architecture with cross-entropy

[30:21](https://youtu.be/AcA8HAYh7IE?list=PLfYUBJiXbdtTIdtE1U8qgyxo4Jy2Y91uj&t=1821) - how to understand  log cross-entropy from scratch

[34:26](https://youtu.be/AcA8HAYh7IE?list=PLfYUBJiXbdtTIdtE1U8qgyxo4Jy2Y91uj&t=2066) - how to write negative log likelihood in pytorch with a trick


[Jump_to lesson 9 video](https://course19.fast.ai/videos/?lesson=9&t=21)


```
#export
from exp.nb_02 import *

def get_data():
    path = datasets.download_data(MNIST_URL, ext='.gz')
    with gzip.open(path, 'rb') as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')
    return map(tensor, (x_train,y_train,x_valid,y_valid))

def normalize(x, m, s): return (x-m)/s
```


```
torch.nn.modules.conv._ConvNd.reset_parameters??
```


```
x_train,y_train,x_valid,y_valid = get_data()
train_mean,train_std = x_train.mean(),x_train.std()
x_train = normalize(x_train, train_mean, train_std)
x_valid = normalize(x_valid, train_mean, train_std)
```


```
x_train = x_train.view(-1,1,28,28)
x_valid = x_valid.view(-1,1,28,28)
x_train.shape,x_valid.shape
```


```
n,*_ = x_train.shape
c = y_train.max()+1
nh = 32
n,c
```


```
l1 = nn.Conv2d(1, nh, 5)
```


```
x = x_valid[:100]
```


```
x.shape
```


```
def stats(x): return x.mean(),x.std()
```


```
l1.weight.shape
```


```
stats(l1.weight),stats(l1.bias)
```


```
t = l1(x)
```


```
stats(t)
```


```
init.kaiming_normal_(l1.weight, a=1.)
stats(l1(x))
```


```
import torch.nn.functional as F
```


```
def f1(x,a=0): return F.leaky_relu(l1(x),a)
```


```
init.kaiming_normal_(l1.weight, a=0)
stats(f1(x))
```


```
l1 = nn.Conv2d(1, nh, 5)
stats(f1(x))
```


```
l1.weight.shape
```


```
# receptive field size
rec_fs = l1.weight[0,0].numel()
rec_fs
```


```
nf,ni,*_ = l1.weight.shape
nf,ni
```


```
fan_in  = ni*rec_fs
fan_out = nf*rec_fs
fan_in,fan_out
```


```
def gain(a): return math.sqrt(2.0 / (1 + a**2))
```


```
gain(1),gain(0),gain(0.01),gain(0.1),gain(math.sqrt(5.))
```


```
torch.zeros(10000).uniform_(-1,1).std()
```


```
1/math.sqrt(3.)
```


```
def kaiming2(x,a, use_fan_out=False):
    nf,ni,*_ = x.shape
    rec_fs = x[0,0].shape.numel()
    fan = nf*rec_fs if use_fan_out else ni*rec_fs
    std = gain(a) / math.sqrt(fan)
    bound = math.sqrt(3.) * std
    x.data.uniform_(-bound,bound)
```


```
kaiming2(l1.weight, a=0);
stats(f1(x))
```


```
kaiming2(l1.weight, a=math.sqrt(5.))
stats(f1(x))
```


```
class Flatten(nn.Module):
    def forward(self,x): return x.view(-1)
```


```
m = nn.Sequential(
    nn.Conv2d(1,8, 5,stride=2,padding=2), nn.ReLU(),
    nn.Conv2d(8,16,3,stride=2,padding=1), nn.ReLU(),
    nn.Conv2d(16,32,3,stride=2,padding=1), nn.ReLU(),
    nn.Conv2d(32,1,3,stride=2,padding=1),
    nn.AdaptiveAvgPool2d(1),
    Flatten(),
)
```


```
y = y_valid[:100].float()
```


```
t = m(x)
stats(t)
```


```
l = mse(t,y)
l.backward()
```


```
stats(m[0].weight.grad)
```


```
init.kaiming_uniform_??
```


```
for l in m:
    if isinstance(l,nn.Conv2d):
        init.kaiming_uniform_(l.weight)
        l.bias.data.zero_()
```


```
t = m(x)
stats(t)
```


```
l = mse(t,y)
l.backward()
stats(m[0].weight.grad)
```

## Export


```
!./notebook2script.py 02a_why_sqrt5.ipynb
```


```
from fastdebug.utils import *
```


```
fastnbs("The forward and backward passes")
```


## <mark style="background-color: #ffff00">the</mark>  <mark style="background-color: #ffff00">forward</mark>  <mark style="background-color: #ffff00">and</mark>  <mark style="background-color: #ffff00">backward</mark>  <mark style="background-color: #FFFF00">passes</mark> 




This section contains only the current heading 2 and its subheadings
### get_data

#### [1:23:03](https://youtu.be/4u8FxNEDUeg?list=PLfYUBJiXbdtTIdtE1U8qgyxo4Jy2Y91uj&t=4983) - how to download and prepare the mnist dataset and wrap the process into a function called `get_data`; 




[Jump_to lesson 8 video](https://course19.fast.ai/videos/?lesson=8&t=4960)

```python
#export
from exp.nb_01 import *

def get_data():
    path = datasets.download_data(MNIST_URL, ext='.gz')
    with gzip.open(path, 'rb') as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')
    return map(tensor, (x_train,y_train,x_valid,y_valid))
```

```python
x_train,y_train,x_valid,y_valid = get_data()
```


### normalize(x, m, s)

#### [1:23:48](https://youtu.be/4u8FxNEDUeg?list=PLfYUBJiXbdtTIdtE1U8qgyxo4Jy2Y91uj&t=5028) - how to create `normalize` function to use broadcast to normalize the Xs and Ys; what does normalization to Xs and Ys mean (make Xs and Ys to have a distribution whose mean is 0 and std is 1)? how to make the mean and std of Xs and Ys to be 0 and 1 (using the formula of normalization below) Why we don't use validation set's mean and std to normalization Xs and Ys of validation set but use those of training set? (make sure validation set and training set share the same scale as training set) What example did Jeremy give to explain the importance of using training set's mean and std for normalization of validation set



```python
def normalize(x, m, s): return (x-m)/s
```

```python
train_mean,train_std = x_train.mean(),x_train.std()
train_mean,train_std
```

```python
x_train = normalize(x_train, train_mean, train_std)
# NB: Use training, not validation mean for validation set
x_valid = normalize(x_valid, train_mean, train_std)
```

### test_near_zero  and  assert

#### [1:24:52](https://youtu.be/4u8FxNEDUeg?list=PLfYUBJiXbdtTIdtE1U8qgyxo4Jy2Y91uj&t=5092) - how to check the mean and std values are close to 0 and 1 using `test_near_zero` using `assert`

```python
train_mean,train_std = x_train.mean(),x_train.std()
train_mean,train_std
```

```python
#export
def test_near_zero(a,tol=1e-3): assert a.abs()<tol, f"Near zero: {a}"
```

```python
test_near_zero(x_train.mean())
test_near_zero(1-x_train.std())
```

### getting dimensions of weights of different layers

#### [1:25:16](https://youtu.be/4u8FxNEDUeg?list=PLfYUBJiXbdtTIdtE1U8qgyxo4Jy2Y91uj&t=5116) - how to get the number of activations of each layer `n` (rows of input), `m` (columns of input), `c` (number of targets/classes) from the shape of `x_train` and `y_train`


```python
n,m = x_train.shape
c = y_train.max()+1
n,m,c
```

start of another heading 2
## Foundations version



[Open `0021_fastai_pt2_2019_fully_connected` in Jupyter Notebook locally](http://localhost:8888/tree/nbs/fastai_notebooks/0021_fastai_pt2_2019_fully_connected.ipynb)



### [1:55:22](https://youtu.be/4u8fxnedueg?list=plfyubjixbdttidte1u8qgyxo4jy2y91uj&t=6922) - how to put <mark style="background-color: #ffff00">forward</mark>  pass <mark style="background-color: #ffff00">and</mark>  <mark style="background-color: #FFFF00">backward</mark>  pass into one function `foward_and_backward`; <mark style="background-color: #ffff00">and</mark>  <mark style="background-color: #FFFF00">backward</mark>  pass is <mark style="background-color: #ffff00">the</mark>  chain rule (people who say no are liars) <mark style="background-color: #ffff00">and</mark>  saving <mark style="background-color: #ffff00">the</mark>  gradients as well; 




This section contains only the current heading 3 and its subheadings

```python
def forward_and_backward(inp, targ):
    # forward pass:
    l1 = inp @ w1 + b1
    l2 = relu(l1)
    out = l2 @ w2 + b2
    # we don't actually need the loss in backward!
    loss = mse(out, targ)
    
    # backward pass:
    mse_grad(out, targ)
    lin_grad(l2, out, w2, b2)
    relu_grad(l1, l2)
    lin_grad(inp, l1, w1, b1)
```

```python
forward_and_backward(x_train, y_train)
```

start of another heading 3
### [1:56:41](https://youtu.be/4u8FxNEDUeg?list=PLfYUBJiXbdtTIdtE1U8qgyxo4Jy2Y91uj&t=7001) - how to use pytorch's gradient calculation functions to test whether our own gradients are calculated correctly; 



[Open `0021_fastai_pt2_2019_fully_connected` in Jupyter Notebook locally](http://localhost:8888/tree/nbs/fastai_notebooks/0021_fastai_pt2_2019_fully_connected.ipynb)



```

```
