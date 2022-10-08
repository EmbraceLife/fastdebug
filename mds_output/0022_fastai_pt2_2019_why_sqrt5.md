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




    (torch.Size([50000, 1, 28, 28]), torch.Size([10000, 1, 28, 28]))




```
n,*_ = x_train.shape
c = y_train.max()+1
nh = 32
n,c
```




    (50000, tensor(10))




```
l1 = nn.Conv2d(1, nh, 5)
```


```
x = x_valid[:100]
```


```
x.shape
```




    torch.Size([100, 1, 28, 28])




```
def stats(x): return x.mean(),x.std()
```


```
l1.weight.shape
```




    torch.Size([32, 1, 5, 5])




```
stats(l1.weight),stats(l1.bias)
```




    ((tensor(-0.0043, grad_fn=<MeanBackward1>),
      tensor(0.1156, grad_fn=<StdBackward0>)),
     (tensor(0.0212, grad_fn=<MeanBackward1>),
      tensor(0.1176, grad_fn=<StdBackward0>)))




```
t = l1(x)
```


```
stats(t)
```




    (tensor(0.0107, grad_fn=<MeanBackward1>),
     tensor(0.5978, grad_fn=<StdBackward0>))




```
init.kaiming_normal_(l1.weight, a=1.)
stats(l1(x))
```




    (tensor(0.0267, grad_fn=<MeanBackward1>),
     tensor(1.1067, grad_fn=<StdBackward0>))




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




    (tensor(0.5547, grad_fn=<MeanBackward1>),
     tensor(1.0199, grad_fn=<StdBackward0>))




```
l1 = nn.Conv2d(1, nh, 5)
stats(f1(x))
```




    (tensor(0.2219, grad_fn=<MeanBackward1>),
     tensor(0.3653, grad_fn=<StdBackward0>))




```
l1.weight.shape
```




    torch.Size([32, 1, 5, 5])




```
# receptive field size
rec_fs = l1.weight[0,0].numel()
rec_fs
```




    25




```
nf,ni,*_ = l1.weight.shape
nf,ni
```




    (32, 1)




```
fan_in  = ni*rec_fs
fan_out = nf*rec_fs
fan_in,fan_out
```




    (25, 800)




```
def gain(a): return math.sqrt(2.0 / (1 + a**2))
```


```
gain(1),gain(0),gain(0.01),gain(0.1),gain(math.sqrt(5.))
```




    (1.0,
     1.4142135623730951,
     1.4141428569978354,
     1.4071950894605838,
     0.5773502691896257)




```
torch.zeros(10000).uniform_(-1,1).std()
```




    tensor(0.5788)




```
1/math.sqrt(3.)
```




    0.5773502691896258




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




    (tensor(0.5603, grad_fn=<MeanBackward1>),
     tensor(1.0921, grad_fn=<StdBackward0>))




```
kaiming2(l1.weight, a=math.sqrt(5.))
stats(f1(x))
```




    (tensor(0.2186, grad_fn=<MeanBackward1>),
     tensor(0.3437, grad_fn=<StdBackward0>))




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




    (tensor(0.0875, grad_fn=<MeanBackward1>),
     tensor(0.0065, grad_fn=<StdBackward0>))




```
l = mse(t,y)
l.backward()
```


```
stats(m[0].weight.grad)
```




    (tensor(0.0054), tensor(0.0333))




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




    (tensor(-0.0352, grad_fn=<MeanBackward1>),
     tensor(0.4043, grad_fn=<StdBackward0>))




```
l = mse(t,y)
l.backward()
stats(m[0].weight.grad)
```




    (tensor(0.0093), tensor(0.4231))



## Export


```
!./notebook2script.py 02a_why_sqrt5.ipynb
```


```

```
