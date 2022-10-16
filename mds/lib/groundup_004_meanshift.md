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

# groundup_004_meanshift

<!-- #raw -->
---
skip_exec: true
---
<!-- #endraw -->

# Clustering


Clustering techniques are unsupervised learning algorithms that try to group unlabelled data into "clusters", using the (typically spatial) structure of the data itself.

The easiest way to demonstrate how clustering works is to simply generate some data and show them in action. We'll start off by importing the libraries we'll be using today.


## Imports

```python
import math, matplotlib.pyplot as plt, operator, torch
from fastdebug.utils import *
```

## torch.manual_seed(1)

```python
torch.manual_seed(1);
```

```python
check(torch.manual_seed)
```

## centroids, torch.randint(low, high, size)

```python
n_clusters=6
n_samples =250
```

To generate our data, we're going to pick 6 random points, which we'll call centroids, and for each point we're going to generate 250 random points about it.

```python
centroids = torch.randint(-35, 35, (n_clusters, 2)).float()
```

```python
centroids
```

```python
check(torch.randint, n=20)
```

## MultivariateNormal, torch.diag, mvn.sample
creating samples on centroids

docs on [MultivariateNormal](https://pytorch.org/docs/stable/distributions.html#multivariatenormal)

```python
from torch.distributions.multivariate_normal import MultivariateNormal
from torch import tensor
```

```python
torch.diag(tensor([5.,5.]))
```

```python
c1 = centroids[0]
c1
```

```python
mvn = MultivariateNormal(c1, torch.diag(tensor([5.,5.])))
c1
mvn.mean
mvn.covariance_matrix
sample = mvn.sample((10,))
```

Suppose there are two data sets X = {3, 2} and Y = {7, 4}. The sample variance of dataset X = 0.5, and Y = 4.5. The covariance between X and Y is 1.5. The covariance matrix is expressed as follows:

```[ 0.5 1.5
  1.5 4.5 ]```

```python
torch.var_mean(sample[:,0]) # calc variance and mean
torch.var_mean(sample[:,1]) 
torch.cov(sample.T) # calc covariance matrix
```

```python
check(mvn.sample)
```

```python
MultivariateNormal(c1, torch.diag(tensor([5.,5.]))).sample((100,)).shape
```

```python
def sample(m): return MultivariateNormal(m, torch.diag(tensor([5.,5.]))).sample((n_samples,))
```

```python
centroids.shape
[c for c in centroids]
slices = [sample(c) for c in centroids]
slices[0].shape
data = torch.cat(slices)
data.shape
```

## plot_centroids_sample, enumerate, plt.scatter, plt.plot
plotting centroids and sample


Below we can see each centroid marked w/ X, and the coloring associated to each respective cluster.

```python
centroids
```

```python
def plot_centroids_sample(centroids, data, n_samples):
    for i, centroid in enumerate(centroids):
        samples = data[i*n_samples:(i+1)*n_samples]
        plt.scatter(samples[:,0], samples[:,1], s=1)
        plt.plot(centroid[0], centroid[1], markersize=10, marker="x", color='k', mew=5)
        plt.plot(centroid[0], centroid[1], markersize=5, marker="x", color='m', mew=2)
```

```python
plot_centroids_sample(centroids, data, n_samples)
```

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

```python
check(torch.linspace)
```

```python
x = torch.linspace(0,10,100)
x.shape
x
```

```python
check(torch.exp)
```

### gaussian kernel in sympy

```python
from sympy import sympify, plot, pi, exp, symbols, sqrt, Eq
```

```python
d, bw, G_1d = symbols("d,bw, G_1d")
```

```python
exp(-0.5*((d/bw))**2)
```

```python
expr = exp(-0.5*((d/bw))**2)/(bw*sqrt(2*pi))
expr
```

```python
Eq(G_1d, expr)
```

```python
expr
expr1 = expr.subs({bw: 2.5})
expr1
```

```python
plot(expr1)
```

### gaussian kernel for weight

```python
def gaussian(d, bw): return torch.exp(-0.5*((d/bw))**2) / (bw*math.sqrt(2*math.pi))
```

```python
x
```

```python
plt.plot(x, gaussian(x,2.5));
```

 This person at the science march certainly remembered!

<img src="http://i.imgur.com/nijQLHw.jpg" width=400>


In our implementation, we choose the bandwidth to be 2.5. 

One easy way to choose bandwidth is to find which bandwidth covers one third of the data.


### from distance to weight

```python
X = data.clone()
x = data[0]
data.shape
X, x
```

```python
sympify("sqrt(a**2 + b**2)")
```

```python
x-X
(x-X)**2
((x-X)**2).sum(1)
torch.sqrt(((x-X)**2).sum(1))
```

```python
dist = torch.sqrt(((x-X)**2).sum(1))
dist.shape
dist.min(), dist.max()
```

```python
weight = gaussian(dist, 2.5)
weight
```

```python
weight.shape,X.shape
weight[:, None].shape
```

```python
(weight[:,None]*X)
```

```python
(weight[:,None]*X).sum(0)
weight.sum()
```

## meanshift

```python
def meanshift(data):
    X = data.clone()
    for it in range(5):
        for i, x in enumerate(X):
            dist = torch.sqrt(((x-X)**2).sum(1))
            weight = gaussian(dist, 2.5)
            X[i] = (weight[:,None]*X).sum(0)/weight.sum()
    return X
```

```python
%time X=meanshift(data)
```

```python
data.shape
X.shape

```

We can see that mean shift clustering has almost reproduced our original clustering. The one exception are the very close clusters, but if we really wanted to differentiate them we could lower the bandwidth.

What is impressive is that this algorithm nearly reproduced the original clusters without telling it how many clusters there should be.

```python
centroids+2, X, n_samples
```

```python
plot_centroids_sample(centroids+2, X, n_samples)
```

All the computation is happening in the <tt>for</tt> loop, which isn't accelerated by pytorch. Each iteration launches a new cuda kernel, which takes time and slows the algorithm down as a whole. Furthermore, each iteration doesn't have enough processing to do to fill up all of the threads of the GPU. But at least the results are correct...

We should be able to accelerate this algorithm with a GPU.


## GPU batched algorithm


To truly accelerate the algorithm, we need to be performing updates on a batch of points per iteration, instead of just one as we were doing.

```python
def dist_b(a,b): return torch.sqrt(((a[None]-b[:,None])**2).sum(2))
```

```python
X=torch.rand(8,2)
x=torch.rand(5,2)
X[None].shape, x[:,None].shape # to make sure broadcast is available
```

```python
dist_b(X, x).shape
```

```python
bs=5
X = data.clone()
x = X[:bs]
weight = gaussian(dist_b(X, x), 2)
weight.shape
```

```python
weight.shape,X.shape
weight[..., None].shape,X[None].shape
```

```python
num = (weight[...,None]*X[None]).sum(1)
num.shape
```

```python
div = weight.sum(1, keepdim=True)
div.shape
```

```python
num/div
```

```python
from fastcore.all import chunked
```

```python
# slice??
```

```python
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

```python editable=false deletable=false run_control={"frozen": true}
data = data.cuda()
```

```python
data
```

```python
meanshift(data)
```

```python
X = meanshift(data).cpu()
```

```python
%timeit -n 1 X = meanshift(data).cpu()
```

```python
plot_centroids_sample(centroids+2, X, n_samples)
```

```python

```
