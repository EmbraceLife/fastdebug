# groundup_004_meanshift
---
skip_exec: true
---
# Clustering

Clustering techniques are unsupervised learning algorithms that try to group unlabelled data into "clusters", using the (typically spatial) structure of the data itself.

The easiest way to demonstrate how clustering works is to simply generate some data and show them in action. We'll start off by importing the libraries we'll be using today.

## Imports


```
import math, matplotlib.pyplot as plt, operator, torch
from fastdebug.utils import *
```


<style>.container { width:100% !important; }</style>


## torch.manual_seed(1)


```
torch.manual_seed(1);
```


```
check(torch.manual_seed)
```

    signature: (seed) -> torch._C.Generator
    __class__: <class 'function'>
    __repr__: <function manual_seed>
    
    __doc__:
    Sets the seed for generating random numbers. Returns a
    `torch.Generator` object.
    
    Args:
        seed (int): The desired seed. Value must be within the inclusive range
            `[-0x8000_0000_0000_0000, 0xffff_ffff_ffff_ffff]`. Otherwise, a RuntimeError
            is raised. Negative inputs are remapped to positive values with the formula
            `0xffff_ffff_ffff_ffff + seed`.
    __dict__: 
    {}
    metaclass: False
    class: False
    decorator: False
    function: True
    method: False


## centroids, torch.randint(low, high, size)


```
n_clusters=6
n_samples =250
```

To generate our data, we're going to pick 6 random points, which we'll call centroids, and for each point we're going to generate 250 random points about it.


```
centroids = torch.randint(-35, 35, (n_clusters, 2)).float()
```


```
centroids
```




    tensor([[  0.,  24.],
            [-31.,   3.],
            [-32., -22.],
            [-14.,  26.],
            [ 14.,  17.],
            [-17.,  34.]])




```
check(torch.randint, n=20)
```

    signature: None
    __class__: <class 'builtin_function_or_method'>
    __repr__: <built-in method randint of type object>
    
    __doc__:
    randint(low=0, high, size, \*, generator=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor
    
    Returns a tensor filled with random integers generated uniformly
    between :attr:`low` (inclusive) and :attr:`high` (exclusive).
    
    The shape of the tensor is defined by the variable argument :attr:`size`.
    
    .. note::
        With the global dtype default (``torch.float32``), this function returns
        a tensor with dtype ``torch.int64``.
    
    Args:
        low (int, optional): Lowest integer to be drawn from the distribution. Default: 0.
        high (int): One above the highest integer to be drawn from the distribution.
        size (tuple): a tuple defining the shape of the output tensor.
    
    Keyword args:
        generator (:class:`torch.Generator`, optional): a pseudorandom number generator for sampling
        out (Tensor, optional): the output tensor.
        dtype (`torch.dtype`, optional) - the desired data type of returned tensor. Default: if ``None``,
            this function returns a tensor with dtype ``torch.int64``.
    __dict__: not exist 
    
    metaclass: False
    class: False
    decorator: False
    function: False
    method: False


## MultivariateNormal, torch.diag, mvn.sample
creating samples on centroids

docs on [MultivariateNormal](https://pytorch.org/docs/stable/distributions.html#multivariatenormal)


```
from torch.distributions.multivariate_normal import MultivariateNormal
from torch import tensor
```


```
torch.diag(tensor([5.,5.]))
```




    tensor([[5., 0.],
            [0., 5.]])




```
c1 = centroids[0]
c1
```




    tensor([ 0., 24.])




```
mvn = MultivariateNormal(c1, torch.diag(tensor([5.,5.])))
c1
mvn.mean
mvn.covariance_matrix
sample = mvn.sample((10,))
```




    tensor([ 0., 24.])






    tensor([ 0., 24.])






    tensor([[5., 0.],
            [0., 5.]])



Suppose there are two data sets X = {3, 2} and Y = {7, 4}. The sample variance of dataset X = 0.5, and Y = 4.5. The covariance between X and Y is 1.5. The covariance matrix is expressed as follows:

```[ 0.5 1.5
  1.5 4.5 ]```


```
torch.var_mean(sample[:,0]) # calc variance and mean
torch.var_mean(sample[:,1]) 
torch.cov(sample.T) # calc covariance matrix
```




    (tensor(6.0844), tensor(1.0054))






    (tensor(1.0135), tensor(23.4095))






    tensor([[ 6.0844, -1.4073],
            [-1.4073,  1.0135]])




```
check(mvn.sample)
```

    signature: (sample_shape=torch.Size([]))
    __class__: <class 'method'>
    __repr__: <bound method Distribution.sample of MultivariateNormal(loc: torch.Size([2]), covariance_matrix: torch.Size([2, 2]))>
    
    __doc__:
    Generates a sample_shape shaped sample or sample_shape shaped batch of
    samples if the distribution parameters are batched.
    __dict__: 
    {}
    metaclass: False
    class: False
    decorator: False
    function: False
    method: True



```
MultivariateNormal(c1, torch.diag(tensor([5.,5.]))).sample((100,)).shape
```




    torch.Size([100, 2])




```
def sample(m): return MultivariateNormal(m, torch.diag(tensor([5.,5.]))).sample((n_samples,))
```


```
centroids.shape
[c for c in centroids]
slices = [sample(c) for c in centroids]
slices[0].shape
data = torch.cat(slices)
data.shape
```




    torch.Size([6, 2])






    [tensor([ 0., 24.]),
     tensor([-31.,   3.]),
     tensor([-32., -22.]),
     tensor([-14.,  26.]),
     tensor([14., 17.]),
     tensor([-17.,  34.])]






    torch.Size([250, 2])






    torch.Size([1500, 2])



## plot_centroids_sample, enumerate, plt.scatter, plt.plot
plotting centroids and sample

Below we can see each centroid marked w/ X, and the coloring associated to each respective cluster.


```
centroids
```




    tensor([[  0.,  24.],
            [-31.,   3.],
            [-32., -22.],
            [-14.,  26.],
            [ 14.,  17.],
            [-17.,  34.]])




```
def plot_centroids_sample(centroids, data, n_samples):
    for i, centroid in enumerate(centroids):
        samples = data[i*n_samples:(i+1)*n_samples]
        plt.scatter(samples[:,0], samples[:,1], s=1)
        plt.plot(centroid[0], centroid[1], markersize=10, marker="x", color='k', mew=5)
        plt.plot(centroid[0], centroid[1], markersize=5, marker="x", color='m', mew=2)
```


```
plot_centroids_sample(centroids, data, n_samples)
```


    
![png](groundup_004_meanshift_files/groundup_004_meanshift_30_0.png)
    


## Mean shift

Most people that have come across clustering algorithms have learnt about **k-means**. Mean shift clustering is a newer and less well-known approach, but it has some important advantages:
* It doesn't require selecting the number of clusters in advance, but instead just requires a **bandwidth** to be specified, which can be easily chosen automatically
* It can handle clusters of any shape, whereas k-means (without using special extensions) requires that clusters be roughly ball shaped.

The algorithm is as follows:
* For each data point x in the sample X, find the distance between that point x and every other point in X
* Create weights for each point in X by using the **Gaussian kernel** of that point's distance to x
    * This weighting approach penalizes points further away from x
    * The rate at which the weights fall to zero is determined by the **bandwidth**, which is the standard deviation of the Gaussian
* Update x as the weighted average of all other points in X, weighted based on the previous step

This will iteratively push points that are close together even closer until they are next to each other.

## gaussian kernel

So here's the definition of the gaussian kernel, which you may remember from high school...

### torch.linspace, torch.exp


```
check(torch.linspace)
```

    signature: None
    __class__: <class 'builtin_function_or_method'>
    __repr__: <built-in method linspace of type object>
    
    __doc__:
    linspace(start, end, steps, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor
    
    Creates a one-dimensional tensor of size :attr:`steps` whose values are evenly
    spaced from :attr:`start` to :attr:`end`, inclusive. That is, the value are:
    
    .. math::
        (\text{start},
        \text{start} + \frac{\text{end} - \text{start}}{\text{steps} - 1},
        \ldots,
        \text{start} + (\text{steps} - 2) * \frac{\text{end} - \text{start}}{\text{steps} - 1},
        \text{end})
    __dict__: not exist 
    
    metaclass: False
    class: False
    decorator: False
    function: False
    method: False



```
x = torch.linspace(0,10,100)
x.shape
x
```




    torch.Size([100])






    tensor([ 0.0000,  0.1010,  0.2020,  0.3030,  0.4040,  0.5051,  0.6061,  0.7071,
             0.8081,  0.9091,  1.0101,  1.1111,  1.2121,  1.3131,  1.4141,  1.5152,
             1.6162,  1.7172,  1.8182,  1.9192,  2.0202,  2.1212,  2.2222,  2.3232,
             2.4242,  2.5253,  2.6263,  2.7273,  2.8283,  2.9293,  3.0303,  3.1313,
             3.2323,  3.3333,  3.4343,  3.5354,  3.6364,  3.7374,  3.8384,  3.9394,
             4.0404,  4.1414,  4.2424,  4.3434,  4.4444,  4.5455,  4.6465,  4.7475,
             4.8485,  4.9495,  5.0505,  5.1515,  5.2525,  5.3535,  5.4545,  5.5556,
             5.6566,  5.7576,  5.8586,  5.9596,  6.0606,  6.1616,  6.2626,  6.3636,
             6.4646,  6.5657,  6.6667,  6.7677,  6.8687,  6.9697,  7.0707,  7.1717,
             7.2727,  7.3737,  7.4747,  7.5758,  7.6768,  7.7778,  7.8788,  7.9798,
             8.0808,  8.1818,  8.2828,  8.3838,  8.4848,  8.5859,  8.6869,  8.7879,
             8.8889,  8.9899,  9.0909,  9.1919,  9.2929,  9.3939,  9.4950,  9.5960,
             9.6970,  9.7980,  9.8990, 10.0000])




```
check(torch.exp)
```

    signature: None
    __class__: <class 'builtin_function_or_method'>
    __repr__: <built-in method exp of type object>
    
    __doc__:
    exp(input, *, out=None) -> Tensor
    
    Returns a new tensor with the exponential of the elements
    of the input tensor :attr:`input`.
    
    .. math::
        y_{i} = e^{x_{i}}
    
    Args:
        input (Tensor): the input tensor.
    
    __dict__: not exist 
    
    metaclass: False
    class: False
    decorator: False
    function: False
    method: False


### gaussian kernel in sympy


```
from sympy import sympify, plot, pi, exp, symbols, sqrt, Eq
```


```
d, bw, G_1d = symbols("d,bw, G_1d")
```


```
exp(-0.5*((d/bw))**2)
```




$\displaystyle e^{- \frac{0.5 d^{2}}{bw^{2}}}$




```
expr = exp(-0.5*((d/bw))**2)/(bw*sqrt(2*pi))
expr
```




$\displaystyle \frac{\sqrt{2} e^{- \frac{0.5 d^{2}}{bw^{2}}}}{2 \sqrt{\pi} bw}$




```
Eq(G_1d, expr)
```




$\displaystyle G_{1d} = \frac{\sqrt{2} e^{- \frac{0.5 d^{2}}{bw^{2}}}}{2 \sqrt{\pi} bw}$




```
expr
expr1 = expr.subs({bw: 2.5})
expr1
```




$\displaystyle \frac{\sqrt{2} e^{- \frac{0.5 d^{2}}{bw^{2}}}}{2 \sqrt{\pi} bw}$






$\displaystyle \frac{0.2 \sqrt{2} e^{- 0.08 d^{2}}}{\sqrt{\pi}}$




```
plot(expr1)
```


    
![png](groundup_004_meanshift_files/groundup_004_meanshift_47_0.png)
    





    <sympy.plotting.plot.Plot>



### gaussian kernel for weight


```
def gaussian(d, bw): return torch.exp(-0.5*((d/bw))**2) / (bw*math.sqrt(2*math.pi))
```


```
x
```




    tensor([ 0.0000,  0.1010,  0.2020,  0.3030,  0.4040,  0.5051,  0.6061,  0.7071,
             0.8081,  0.9091,  1.0101,  1.1111,  1.2121,  1.3131,  1.4141,  1.5152,
             1.6162,  1.7172,  1.8182,  1.9192,  2.0202,  2.1212,  2.2222,  2.3232,
             2.4242,  2.5253,  2.6263,  2.7273,  2.8283,  2.9293,  3.0303,  3.1313,
             3.2323,  3.3333,  3.4343,  3.5354,  3.6364,  3.7374,  3.8384,  3.9394,
             4.0404,  4.1414,  4.2424,  4.3434,  4.4444,  4.5455,  4.6465,  4.7475,
             4.8485,  4.9495,  5.0505,  5.1515,  5.2525,  5.3535,  5.4545,  5.5556,
             5.6566,  5.7576,  5.8586,  5.9596,  6.0606,  6.1616,  6.2626,  6.3636,
             6.4646,  6.5657,  6.6667,  6.7677,  6.8687,  6.9697,  7.0707,  7.1717,
             7.2727,  7.3737,  7.4747,  7.5758,  7.6768,  7.7778,  7.8788,  7.9798,
             8.0808,  8.1818,  8.2828,  8.3838,  8.4848,  8.5859,  8.6869,  8.7879,
             8.8889,  8.9899,  9.0909,  9.1919,  9.2929,  9.3939,  9.4950,  9.5960,
             9.6970,  9.7980,  9.8990, 10.0000])




```
plt.plot(x, gaussian(x,2.5));
```


    
![png](groundup_004_meanshift_files/groundup_004_meanshift_51_0.png)
    


 This person at the science march certainly remembered!

<img src="http://i.imgur.com/nijQLHw.jpg" width=400>

In our implementation, we choose the bandwidth to be 2.5. 

One easy way to choose bandwidth is to find which bandwidth covers one third of the data.

### from distance to weight


```
X = data.clone()
x = data[0]
data.shape
X, x
```




    torch.Size([1500, 2])






    (tensor([[ -2.2934,  25.1656],
             [ -1.0131,  23.7182],
             [  5.2227,  22.6092],
             ...,
             [-17.3308,  33.4929],
             [-12.5499,  33.5847],
             [-16.5074,  35.8346]]),
     tensor([-2.2934, 25.1656]))




```
sympify("sqrt(a**2 + b**2)")
```




$\displaystyle \sqrt{a^{2} + b^{2}}$




```
x-X
(x-X)**2
((x-X)**2).sum(1)
torch.sqrt(((x-X)**2).sum(1))
```




    tensor([[  0.0000,   0.0000],
            [ -1.2803,   1.4474],
            [ -7.5161,   2.5564],
            ...,
            [ 15.0373,  -8.3273],
            [ 10.2565,  -8.4191],
            [ 14.2140, -10.6690]])






    tensor([[  0.0000,   0.0000],
            [  1.6393,   2.0948],
            [ 56.4922,   6.5353],
            ...,
            [226.1211,  69.3436],
            [105.1960,  70.8818],
            [202.0365, 113.8286]])






    tensor([  0.0000,   3.7341,  63.0275,  ..., 295.4647, 176.0777, 315.8650])






    tensor([ 0.0000,  1.9324,  7.9390,  ..., 17.1891, 13.2694, 17.7726])




```
dist = torch.sqrt(((x-X)**2).sum(1))
dist.shape
dist.min(), dist.max()
```




    torch.Size([1500])






    (tensor(0.), tensor(61.8455))




```
weight = gaussian(dist, 2.5)
weight
```




    tensor([1.5958e-01, 1.1837e-01, 1.0308e-03,  ..., 8.6591e-12, 1.2173e-07,
            1.6932e-12])




```
weight.shape,X.shape
weight[:, None].shape
```




    (torch.Size([1500]), torch.Size([1500, 2]))






    torch.Size([1500, 1])




```
(weight[:,None]*X)
```




    tensor([[-3.6598e-01,  4.0158e+00],
            [-1.1992e-01,  2.8075e+00],
            [ 5.3835e-03,  2.3305e-02],
            ...,
            [-1.5007e-10,  2.9002e-10],
            [-1.5277e-06,  4.0883e-06],
            [-2.7950e-11,  6.0673e-11]])




```
(weight[:,None]*X).sum(0)
weight.sum()
```




    tensor([-18.4532, 425.6993])






    tensor(17.3609)



## meanshift


```
def meanshift(data):
    X = data.clone()
    for it in range(5):
        for i, x in enumerate(X):
            dist = torch.sqrt(((x-X)**2).sum(1))
            weight = gaussian(dist, 2.5)
            X[i] = (weight[:,None]*X).sum(0)/weight.sum()
    return X
```


```
%time X=meanshift(data)
```

    CPU times: user 277 ms, sys: 1 ms, total: 278 ms
    Wall time: 278 ms



```
data.shape
X.shape

```




    torch.Size([1500, 2])






    torch.Size([1500, 2])



We can see that mean shift clustering has almost reproduced our original clustering. The one exception are the very close clusters, but if we really wanted to differentiate them we could lower the bandwidth.

What is impressive is that this algorithm nearly reproduced the original clusters without telling it how many clusters there should be.


```
centroids+2, X, n_samples
```




    (tensor([[  2.,  26.],
             [-29.,   5.],
             [-30., -20.],
             [-12.,  28.],
             [ 16.,  19.],
             [-15.,  36.]]),
     tensor([[ -0.1667,  24.2604],
             [ -0.1667,  24.2604],
             [ -0.1667,  24.2604],
             ...,
             [-15.4105,  30.0576],
             [-15.4081,  30.0514],
             [-15.4071,  30.0487]]),
     250)




```
plot_centroids_sample(centroids+2, X, n_samples)
```


    
![png](groundup_004_meanshift_files/groundup_004_meanshift_69_0.png)
    


All the computation is happening in the <tt>for</tt> loop, which isn't accelerated by pytorch. Each iteration launches a new cuda kernel, which takes time and slows the algorithm down as a whole. Furthermore, each iteration doesn't have enough processing to do to fill up all of the threads of the GPU. But at least the results are correct...

We should be able to accelerate this algorithm with a GPU.

## GPU batched algorithm

To truly accelerate the algorithm, we need to be performing updates on a batch of points per iteration, instead of just one as we were doing.


```
def dist_b(a,b): return torch.sqrt(((a[None]-b[:,None])**2).sum(2))
```


```
X=torch.rand(8,2)
x=torch.rand(5,2)
X[None].shape, x[:,None].shape # to make sure broadcast is available
```




    (torch.Size([1, 8, 2]), torch.Size([5, 1, 2]))




```
dist_b(X, x).shape
```




    torch.Size([5, 8])




```
bs=5
X = data.clone()
x = X[:bs]
weight = gaussian(dist_b(X, x), 2)
weight.shape
```




    torch.Size([5, 1500])




```
weight.shape,X.shape
weight[..., None].shape,X[None].shape
```




    (torch.Size([5, 1500]), torch.Size([1500, 2]))






    (torch.Size([5, 1500, 1]), torch.Size([1, 1500, 2]))




```
num = (weight[...,None]*X[None]).sum(1)
num.shape
```




    torch.Size([5, 2])




```
div = weight.sum(1, keepdim=True)
div.shape
```




    torch.Size([5, 1])




```
num/div
```




    tensor([[-1.2688, 24.6079],
            [-0.5848, 23.9076],
            [ 2.8672, 23.3891],
            [-0.3928, 22.5619],
            [-1.2959, 23.5955]])




```
from fastcore.all import chunked
```


```
slice??
```


```
slice(i, min(4,6))
```


      Input In [59]
        slice(i, min(4,6)
                         ^
    SyntaxError: unexpected EOF while parsing




```
def meanshift(data, bs=500):
    n = len(data)
    X = data.clone()
    for it in range(5):
        for i in range(0, n, bs):
            s = slice(i, min(i+bs,n))
            weight = gaussian(dist_b(X, X[s]), 2)
            num = (weight[...,None]*X[None]).sum(1)
            div = weight.sum(1, keepdim=True)
            X[s] = num/div
    return X
```

Although each iteration still has to launch a new cuda kernel, there are now fewer iterations, and the acceleration from updating a batch of points more than makes up for it.


```
data = data.cuda()
```


```
data
```




    tensor([[ -1.3219,  24.6122],
            [ -2.1575,  23.4727],
            [ -1.5585,  21.4045],
            ...,
            [-14.2147,  31.9600],
            [-14.6337,  33.5766],
            [-17.4343,  37.4739]], device='cuda:0')




```
meanshift(data)
```




    tensor([[  0.2856,  24.2307],
            [  0.2856,  24.2307],
            [  0.2856,  24.2307],
            ...,
            [-17.1203,  34.0656],
            [-17.1203,  34.0656],
            [-17.1203,  34.0656]], device='cuda:0')




```
X = meanshift(data).cpu()
```


```
%timeit -n 1 X = meanshift(data).cpu()
```

    4.5 ms ± 171 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)



```
plot_data(centroids+2, X, n_samples)
```


    
![png](groundup_004_meanshift_files/groundup_004_meanshift_91_0.png)
    



```

```
