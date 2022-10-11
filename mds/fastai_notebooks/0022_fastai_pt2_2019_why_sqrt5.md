---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.0
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# 0022_fastai_pt2_2019_why_sqrt5

<!-- #raw -->
---
skip_exec: true
---
<!-- #endraw -->

```python
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

```python
#export
from exp.nb_02 import *

def get_data():
    path = datasets.download_data(MNIST_URL, ext='.gz')
    with gzip.open(path, 'rb') as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')
    return map(tensor, (x_train,y_train,x_valid,y_valid))

def normalize(x, m, s): return (x-m)/s
```

```python
torch.nn.modules.conv._ConvNd.reset_parameters??
```

```python
x_train,y_train,x_valid,y_valid = get_data()
train_mean,train_std = x_train.mean(),x_train.std()
x_train = normalize(x_train, train_mean, train_std)
x_valid = normalize(x_valid, train_mean, train_std)
```

```python
x_train = x_train.view(-1,1,28,28)
x_valid = x_valid.view(-1,1,28,28)
x_train.shape,x_valid.shape
```

```python
n,*_ = x_train.shape
c = y_train.max()+1
nh = 32
n,c
```

```python
l1 = nn.Conv2d(1, nh, 5)
```

```python
x = x_valid[:100]
```

```python
x.shape
```

```python
def stats(x): return x.mean(),x.std()
```

```python
l1.weight.shape
```

```python
stats(l1.weight),stats(l1.bias)
```

```python
t = l1(x)
```

```python
stats(t)
```

```python
init.kaiming_normal_(l1.weight, a=1.)
stats(l1(x))
```

```python
import torch.nn.functional as F
```

```python
def f1(x,a=0): return F.leaky_relu(l1(x),a)
```

```python
init.kaiming_normal_(l1.weight, a=0)
stats(f1(x))
```

```python
l1 = nn.Conv2d(1, nh, 5)
stats(f1(x))
```

```python
l1.weight.shape
```

```python
# receptive field size
rec_fs = l1.weight[0,0].numel()
rec_fs
```

```python
nf,ni,*_ = l1.weight.shape
nf,ni
```

```python
fan_in  = ni*rec_fs
fan_out = nf*rec_fs
fan_in,fan_out
```

```python
def gain(a): return math.sqrt(2.0 / (1 + a**2))
```

```python
gain(1),gain(0),gain(0.01),gain(0.1),gain(math.sqrt(5.))
```

```python
torch.zeros(10000).uniform_(-1,1).std()
```

```python
1/math.sqrt(3.)
```

```python
def kaiming2(x,a, use_fan_out=False):
    nf,ni,*_ = x.shape
    rec_fs = x[0,0].shape.numel()
    fan = nf*rec_fs if use_fan_out else ni*rec_fs
    std = gain(a) / math.sqrt(fan)
    bound = math.sqrt(3.) * std
    x.data.uniform_(-bound,bound)
```

```python
kaiming2(l1.weight, a=0);
stats(f1(x))
```

```python
kaiming2(l1.weight, a=math.sqrt(5.))
stats(f1(x))
```

```python
class Flatten(nn.Module):
    def forward(self,x): return x.view(-1)
```

```python
m = nn.Sequential(
    nn.Conv2d(1,8, 5,stride=2,padding=2), nn.ReLU(),
    nn.Conv2d(8,16,3,stride=2,padding=1), nn.ReLU(),
    nn.Conv2d(16,32,3,stride=2,padding=1), nn.ReLU(),
    nn.Conv2d(32,1,3,stride=2,padding=1),
    nn.AdaptiveAvgPool2d(1),
    Flatten(),
)
```

```python
y = y_valid[:100].float()
```

```python
t = m(x)
stats(t)
```

```python
l = mse(t,y)
l.backward()
```

```python
stats(m[0].weight.grad)
```

```python
init.kaiming_uniform_??
```

```python
for l in m:
    if isinstance(l,nn.Conv2d):
        init.kaiming_uniform_(l.weight)
        l.bias.data.zero_()
```

```python
t = m(x)
stats(t)
```

```python
l = mse(t,y)
l.backward()
stats(m[0].weight.grad)
```

## Export

```python
!./notebook2script.py 02a_why_sqrt5.ipynb
```

```python
from fastdebug.utils import *
```

```python
fastnbs("The forward and backward passes")
```

```python

```
