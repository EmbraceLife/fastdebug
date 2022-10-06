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

# 0005_fastai_linear_neuralnet_scratch


**Official course site**:  for lesson [3](https://course.fast.ai/Lessons/lesson3.html)    

**Official notebooks** [repo](https://github.com/fastai/course22), on [nbviewer](https://nbviewer.org/github/fastai/course22/tree/master/)

Official **neuralnet from scratch** [notebook](https://www.kaggle.com/code/jhoward/linear-model-and-neural-net-from-scratch) on kaggle     



## how to not execute the entire notebook

<!-- #raw -->
---
skip_exec: true
---
<!-- #endraw -->

## Introduction


In this notebook we're going to build and train a deep learning model "from scratch" -- by which I mean that we're not going to use any pre-built architecture, or optimizers, or data loading frameworks, etc.

We'll be assuming you already know the basics of how a neural network works. If you don't, read this notebook first: [How does a neural net really work?
](https://www.kaggle.com/code/jhoward/how-does-a-neural-net-really-work). We'll be using Kaggle's [Titanic](https://www.kaggle.com/competitions/titanic/) competition in this notebook, because it's very small and simple, but also has displays many of the tricky real-life issues that we need to handle in most practical projects. (Note, however, that this competition is a small "learner" competition on Kaggle, so don't expect to actually see much benefits from using a neural net just yet; that will come once we try our some real competitions!)

It's great to be able to run the same notebook on your own machine or Colab, as well as Kaggle. To allow for this, we use this code to download the data as needed when not on Kaggle (see [this notebook](https://www.kaggle.com/code/jhoward/getting-started-with-nlp-for-absolute-beginners/) for details about this technique):


## How to download kaggle dataset to your local machine or colab? how to ues kaggle api and zipfile to download data into specified folder; how to use `pathlib.Path` to create a path;

```python
import os
from pathlib import Path

iskaggle = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '')
if iskaggle: path = Path('../input/titanic')
else:
    path = Path('titanic')
    if not path.exists():
        import zipfile,kaggle
        kaggle.api.competition_download_cli(str(path))
        zipfile.ZipFile(f'{path}.zip').extractall(path)
```

Note that the data for Kaggle comps always lives in the `../input` folder. The easiest way to get the path is to click the "K" button in the top-right of the Kaggle notebook, click on the folder shown there, and click the copy button.

We'll be using *numpy* and *pytorch* for array calculations in this notebook, and *pandas* for working with tabular data, so we'll import them and set them to display using a bit more space than they default to.


## how to set the print display option for numpy, torch and pandas

```python
import torch, numpy as np, pandas as pd
np.set_printoptions(linewidth=140)
torch.set_printoptions(linewidth=140, sci_mode=False, edgeitems=7)
pd.set_option('display.width', 140)
```

## Cleaning the data


### how to read csv file with pandas and `path/'subfolder_name'`


This is a *tabular data* competition -- the data is in the form of a table. It's provided as a Comma Separated Values (CSV) file. We can open it using the *pandas* library, which will create a `DataFrame`.

```python
df = pd.read_csv(path/'train.csv')
df
```

### why missing value is a problem? how to find out the num of missing values of each column with pandas?


As we learned in the *How does a neural net really work* notebook, we going to want to multiply each column by some coefficients. But we can see in the `Cabin` column that there are `NaN` values, which is how Pandas refers to missing values. We can't multiply something by a missing value!

Let's check which columns contain `NaN` values. Pandas' `isna()` function returns `True` (which is treated as `1` when used as a number) for `NaN` values, so we can just add them up for each column:

```python
df.isna().sum()
```

### which value is most used to replace missing value? how to get mode for each column with pandas using `iloc[0]`


Notice that by default Pandas sums over columns.

We'll need to replace the missing values with something. It doesn't generally matter too much what we choose. We'll use the most common value (the "*mode*"). We can use the `mode` function for that. One wrinkle is that it returns more than one row in the case of ties, so we just grab the first row with `iloc[0]`:

```python
modes = df.mode().iloc[0]
modes
```

### how to use pandas `iloc` function


BTW, it's never a good idea to use functions without understanding them. So be sure to google for anything you're not familiar with. E.g if you want to learn about `iloc` (which is a very important function indeed!) then Google will give you a link to a [great tutorial](https://www.shanelynn.ie/pandas-iloc-loc-select-rows-and-columns-dataframe/).


### how to fill missing values with mode without making a new copy with `pandas.fillna`


Now that we've got the mode of each column, we can use `fillna` to replace the missing values with the mode of each column. We'll do it "in place" -- meaning that we'll change the dataframe itself, rather than returning a new one.

```python
df.fillna(modes, inplace=True)
```

We can now check there's no missing values left:

```python
df.isna().sum()
```

### how to get a quick summary of all the numeric columns with pandas and numpy


Here's how we get a quick summary of all the numeric columns in the dataset:

```python
import numpy as np

df.describe(include=(np.number))
```

### what is long-tailed data in histogram and why it is a problem for neuralnet


We can see that `Fare` contains mainly values of around `0` to `30`, but there's a few really big ones. This is very common with fields contain monetary values, and it can cause problems for our model, because once that column is multiplied by a coefficient later, the few rows with really big values will dominate the result.

You can see the issue most clearly visually by looking at a histogram, which shows a long tail to the right (and don't forget: if you're not entirely sure what a histogram is, Google "[histogram tutorial](https://www.google.com/search?q=histogram+tutorial&oq=histogram+tutorial)" and do a bit of reading before continuing on):


### how to plot histogram with pandas on a single column

```python
df['Fare'].hist();
```

### how to fix long-tailed data with logarithm; why should logarithm work; how to handle zero values when applying logarithm


To fix this, the most common approach is to take the logarithm, which squishes the big numbers and makes the distribution more reasonable. Note, however, that there are zeros in the `Fare` column, and `log(0)` is infinite -- to fix this, we'll simply add `1` to all values first:

```python
df['LogFare'] = np.log(df['Fare']+1)
```

The histogram now shows a more even distribution of values without the long tail:

```python
df['LogFare'].hist();
```

It looks from the `describe()` output like `Pclass` contains just 3 values, which we can confirm by looking at the [Data Dictionary](https://www.kaggle.com/competitions/titanic/data) (which you should always study carefully for any project!) -- 

```python
pclasses = sorted(df.Pclass.unique())
pclasses
```

### how to get a quick summary of all the non-numeric columns with pandas


Here's how we get a quick summary of all the non-numeric columns in the dataset:

```python
df.describe(include=[object])
```

### when do we need dummy variables and how to create dummy variables with pandas


Clearly we can't multiply strings like `male` or `S` by coefficients, so we need to replace those with numbers.

We do that by creating new columns containing *dummy variables*. A dummy variable is a column that contains a `1` where a particular column contains a particular value, or a `0` otherwise. For instance, we could create a dummy variable for `Sex='male'`, which would be a new column containing `1` for rows where `Sex` is `'male'`, and 0 for rows where it isn't.

Pandas can create these automatically using `get_dummies`, which also remove the original columns. We'll create dummy variables for `Pclass`, even although it's numeric, since the numbers `1`, `2`, and `3` correspond to first, second, and third class cabins - not to counts or measures that make sense to multiply by. We'll also create dummies for `Sex` and `Embarked` since we'll want to use those as predictors in our model. On the other hand, `Cabin`, `Name`, and `Ticket` have too many unique values for it to make sense creating dummy variables for them.

```python
df = pd.get_dummies(df, columns=["Sex","Pclass","Embarked"])
df.columns
```

### how to check the first few rows of selected columns with pandas


We can see that 5 columns have been added to the end -- one for each of the possible values of each of the three columns we requested, and that those three requested columns have been removed.

Here's what the first few rows of those newly added columns look like:

```python
added_cols = ['Sex_male', 'Sex_female', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Embarked_C', 'Embarked_Q', 'Embarked_S']
df[added_cols].head()
```

### how to create dependent/target variable and independent/predictor variables in PyTorch tensors; how to create variables in tensor from pandas dataframe



Now we can create our independent (predictors) and dependent (target) variables. They both need to be PyTorch tensors. Our dependent variable is `Survived`:

```python
from torch import tensor

t_dep = tensor(df.Survived)
```

Our independent variables are all the continuous variables of interest plus all the dummy variables we just created:

```python
indep_cols = ['Age', 'SibSp', 'Parch', 'LogFare'] + added_cols

t_indep = tensor(df[indep_cols].values, dtype=torch.float)
t_indep
```

### how to check the size (rows and columns) of independent variables in tensor


Here's the number of rows and columns we have for our independent variables:

```python
t_indep.shape
```

## Setting up a linear model


### how to create coefficients for each (column) of our independent variables; how to get random seed in torch; how to get the num of columns; how to create random number between -0.5 and 0.5;


Now that we've got a matrix of independent variables and a dependent variable vector, we can work on calculating our predictions and our loss. In this section, we're going to manually do a single step of calculating predictions and loss for every row of our data.

Our first model will be a simple linear model. We'll need a coefficient for each column in `t_indep`. We'll pick random numbers in the range `(-0.5,0.5)`, and set our manual seed so that my explanations in the prose in this notebook will be consistent with what you see when you run it.

```python
torch.manual_seed(442)
n_coeff = t_indep.shape[1]
coeffs = torch.rand(n_coeff)-0.5
coeffs
```

### why no bias or a constant is needed for this Titanic dataset?


Our predictions will be calculated by multiplying each row by the coefficients, and adding them up. One interesting point here is that we don't need a separate constant term (also known as a "bias" or "intercept" term), or a column of all `1`s to give the same effect has having a constant term. That's because our dummy variables already cover the entire dataset -- e.g. there's a column for "male" and a column for "female", and everyone in the dataset is in exactly one of these; therefore, we don't need a separate intercept term to cover rows that aren't otherwise part of a column.

Here's what the multiplication looks like:

```python
t_indep*coeffs
```

### why a column `Age` having higher values than other columns can cause problem for our model; how to solve this problem by making them the same scale; how to get the max value of each column with pandas dataframe max func


We can see we've got a problem here. The sums of each row will be dominated by the first column, which is `Age`, since that's bigger on average than all the others.

Let's make all the columns contain numbers from `0` to `1`, by dividing each column by its `max()`:

```python
vals,indices = t_indep.max(dim=0)
t_indep = t_indep / vals
```

As we see, that removes the problem of one column dominating all the others:

```python
t_indep*coeffs
```

### what is maxtrix by vector operation (multiply or divide)


One thing you hopefully noticed is how amazingly cool this line of code is:

    t_indep = t_indep / vals

That is dividing a matrix by a vector -- what on earth does that mean?!? The trick here is that we're taking advantage of a technique in numpy and PyTorch (and many other languages, going all the way back to APL) called [broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html). In short, this acts as if there's a separate copy of the vector for every row of the matrix, so it divides each row of the matrix by the vector. In practice, it doesn't actually make any copies, and does the whole thing in a highly optimized way, taking full advantage of modern CPUs (or, indeed, GPUs, if we're using them). Broadcasting is one of the most important techniques for making your code concise, maintainable, and fast, so it's well worth studying and practicing.


### How to calculate the prediction of a linear model


We can now create predictions from our linear model, by adding up the rows of the product:

```python
preds = (t_indep*coeffs).sum(axis=1)
```

### how to look at the first 10 values of predictions


Let's take a look at the first few:

```python
preds[:10]
```

### how to calc mean absolute error


Of course, these predictions aren't going to be any use, since our coefficients are random -- they're just a starting point for our gradient descent process.

To do gradient descent, we need a loss function. Taking the average error of the rows (i.e. the absolute value of the difference between the prediction and the dependent) is generally a reasonable approach:

```python
loss = torch.abs(preds-t_dep).mean()
loss
```

### how to calc predictions with a func `calc_preds`; how to calc loss with a func `calc_loss`


Now that we've tested out a way of calculating predictions, and loss, let's pop them into functions to make life easier:

```python
def calc_preds(coeffs, indeps): return (indeps*coeffs).sum(axis=1)
def calc_loss(coeffs, indeps, deps): return torch.abs(calc_preds(coeffs, indeps)-deps).mean()
```

## Doing a gradient descent step


In this section, we're going to do a single "epoch" of gradient descent manually. The only thing we're going to automate is calculating gradients, because let's face it that's pretty tedious and entirely pointless to do by hand! To get PyTorch to calculate gradients, we'll need to call `requires_grad_()` on our `coeffs` (if you're not sure why, review the previous notebook, [How does a neural net really work?](https://www.kaggle.com/code/jhoward/how-does-a-neural-net-really-work), before continuing):

```python
coeffs.requires_grad_()
```

Now when we calculate our loss, PyTorch will keep track of all the steps, so we'll be able to get the gradients afterwards:

```python
loss = calc_loss(coeffs, t_indep, t_dep)
loss
```

Use `backward()` to ask PyTorch to calculate gradients now:

```python
loss.backward()
```

Let's see what they look like:

```python
coeffs.grad
```

### How to cacl gradients for coefficients


Note that each time we call `backward`, the gradients are actually *added* to whatever is in the `.grad` attribute. Let's try running the above steps again:

```python
loss = calc_loss(coeffs, t_indep, t_dep)
loss.backward()
coeffs.grad
```

### why set gradients to zero after each gradient descent step; how to set gradient to zero; how to do one iteration of training


As you see, our `.grad` values are have doubled. That's because it added the gradients a second time. For this reason, after we use the gradients to do a gradient descent step, we need to set them back to zero.

We can now do one gradient descent step, and check that our loss decreases:

```python
loss = calc_loss(coeffs, t_indep, t_dep)
loss.backward()
with torch.no_grad():
    coeffs.sub_(coeffs.grad * 0.1)
    coeffs.grad.zero_()
    print(calc_loss(coeffs, t_indep, t_dep))
```

### what does _ mean for `coeffs.sub_()` and `grad.zero_()`


Note that `a.sub_(b)` subtracts `b` from `a` in-place. In PyTorch, any method that ends in `_` changes its object in-place. Similarly, `a.zero_()` sets all elements of a tensor to zero.


## Training the linear model


### how to split the dataset by using train and valid idx produced by `fastai.data.transforms.RandomSplitter`


Before we begin training our model, we'll need to ensure that we hold out a validation set for calculating our metrics (for details on this, see "[Getting started with NLP for absolute beginners](https://www.kaggle.com/code/jhoward/getting-started-with-nlp-for-absolute-beginners#Test-and-validation-sets)".

There's lots of different ways we can do this. In the next notebook we'll be comparing our approach here to what the fastai library does, so we'll want to ensure we split the data in the same way. So let's use `RandomSplitter` to get indices that will split our data into training and validation sets:

```python
from fastai.data.transforms import RandomSplitter
trn_split,val_split=RandomSplitter(seed=42)(df)
```

Now we can apply those indicies to our independent and dependent variables:

```python
trn_indep,val_indep = t_indep[trn_split],t_indep[val_split]
trn_dep,val_dep = t_dep[trn_split],t_dep[val_split]
len(trn_indep),len(val_indep)
```

We'll create functions for the three things we did manually above: updating `coeffs`, doing one full gradient descent step, and initilising `coeffs` to random numbers:


### how to udpate coefficients in a function `update_coeffs`

```python
def update_coeffs(coeffs, lr):
    coeffs.sub_(coeffs.grad * lr)
    coeffs.grad.zero_()
```

### how to do one epoch training in a function `one_epoch`

```python
def one_epoch(coeffs, lr):
    loss = calc_loss(coeffs, trn_indep, trn_dep)
    loss.backward()
    with torch.no_grad(): update_coeffs(coeffs, lr)
    print(f"{loss:.3f}", end="; ")
```

### how to initializing coefficients in a function `init_coeffs`

```python
def init_coeffs(): return (torch.rand(n_coeff)-0.5).requires_grad_()
```

### how to integrate funcs above to form a function `train_model` on multiple epochs


We can now use these functions to train our model:

```python
def train_model(epochs=30, lr=0.01):
    torch.manual_seed(442)
    coeffs = init_coeffs()
    for i in range(epochs): one_epoch(coeffs, lr=lr)
    return coeffs
```

Let's try it. Our loss will print at the end of every step, so we hope we'll see it going down:

```python
coeffs = train_model(18, lr=0.2)
```

### how to display coefficients of the model with func `show_coeffs`


It does!

Let's take a look at the coefficients for each column:

```python
def show_coeffs(): return dict(zip(indep_cols, coeffs.requires_grad_(False)))
show_coeffs()
```

## Measuring accuracy


### There are many possible loss options such as accuracy other than mean absolute error


The Kaggle competition is not, however, scored by absolute error (which is our loss function). It's scored by *accuracy* -- the proportion of rows where we correctly predict survival. Let's see how accurate we were on the validation set. First, calculate the predictions:

```python
preds = calc_preds(coeffs, val_indep)
```

### how to calc accuracy for the binary dependent variable


We'll assume that any passenger with a score of over `0.5` is predicted to survive. So that means we're correct for each row where `preds>0.5` is the same as the dependent variable:

```python
results = val_dep.bool()==(preds>0.5)
results[:16]
```

Let's see what our average accuracy is:

```python
results.float().mean()
```

### how to wrap the process of calc accuracy using coeffs into a func `acc(coeffs)`


That's not a bad start at all! We'll create a function so we can calcuate the accuracy easy for other models we train:

```python
def acc(coeffs): return (val_dep.bool()==(calc_preds(coeffs, val_indep)>0.5)).float().mean()
acc(coeffs)
```

## Using sigmoid


### when will we be needing something like sigmoid


Looking at our predictions, there's one obvious problem -- some of our predictions of the probability of survival are `>1`, and some are `<0`:

```python
preds[:28]
```

### how to write and plot a func like `sigmoid` using sympy


To fix this, we should pass every prediction through the *sigmoid function*, which has a minimum at zero and maximum at one, and is defined as follows:

```python
import sympy
sympy.plot("1/(1+exp(-x))", xlim=(-5,5));
```

### how to update `calc_preds` by wrapping `torch.sigmoid` around prediction


PyTorch already defines that function for us, so we can modify `calc_preds` to use it:

```python
def calc_preds(coeffs, indeps): return torch.sigmoid((indeps*coeffs).sum(axis=1))
```

Let's train a new model now, using this updated function to calculate predictions:

```python
coeffs = train_model(lr=100)
```

The loss has improved by a lot. Let's check the accuracy:

```python
acc(coeffs)
```

That's improved too! Here's the coefficients of our trained model:

```python
show_coeffs()
```

These coefficients seem reasonable -- in general, older people and males were less likely to survive, and first class passengers were more likely to survive.


## Submitting to Kaggle


Now that we've got a trained model, we can prepare a submission to Kaggle. To do that, first we need to read the test set:


### read test data using `pandas.read_csv`

```python
tst_df = pd.read_csv(path/'test.csv')
```

### why and how to fill the missing value in Fare column with 0 instead of mode


In this case, it turns out that the test set is missing `Fare` for one passenger. We'll just fill it with `0` to avoid problems:

```python
tst_df['Fare'] = tst_df.Fare.fillna(0)
```

### how to handle missing values, long-tailed distribution and dummies together for test data


Now we can just copy the same steps we did to our training set and do the same exact things on our test set to preprocess the data:

```python
tst_df.fillna(modes, inplace=True)
tst_df['LogFare'] = np.log(tst_df['Fare']+1)
tst_df = pd.get_dummies(tst_df, columns=["Sex","Pclass","Embarked"])
```

### how to turn independent variable values into tensor

```python
tst_indep = tensor(tst_df[indep_cols].values, dtype=torch.float)
```

### how to make sure independent variable in test data share the same value scare with those in training data

```python
tst_indep = tst_indep / vals
```

### how to turn true or false into 1 or 0 and save them into a column


Let's calculate our predictions of which passengers survived in the test set:

```python
tst_df['Survived'] = (calc_preds(tst_indep, coeffs)>0.5).int()
```

### how to select two columns of a dataframe and save them into a csv file using `to_csv`


The sample submission on the Kaggle competition site shows that we're expected to upload a CSV with just `PassengerId` and `Survived`, so let's create that and save it:

```python
sub_df = tst_df[['PassengerId','Survived']]
sub_df.to_csv('sub.csv', index=False)
```

### how to check the first few lines of the csv file using `!head`


We can check the first few rows of the file to make sure it looks reasonable:

```python
!head sub.csv
```

When you click "save version" in Kaggle, and wait for the notebook to run, you'll see that `sub.csv` appears in the "Data" tab. Clicking on that file will show a *Submit* button, which allows you to submit to the competition.


## Using matrix product


### how to do matrix product `@` between a matrix and a vector with PyTorch; how to use `@` instead of doing multiplication and then addition together

We can make things quite a bit neater...

Take a look at the inner-most calculation we're doing to get the predictions:

```python
(val_indep*coeffs).sum(axis=1)
```

Multiplying elements together and then adding across rows is identical to doing a matrix-vector product! Python uses the `@` operator to indicate matrix products, and is supported by PyTorch tensors. Therefore, we can replicate the above calculate more simply like so:

```python
val_indep@coeffs
```

### update `calc_preds` func using matrix multiplication `@`


It also turns out that this is much faster, because matrix products in PyTorch are very highly optimised.

Let's use this to replace how `calc_preds` works:

```python
def calc_preds(coeffs, indeps): return torch.sigmoid(indeps@coeffs)
```

### how to initialize coeffs and turn it into a matrix with a single column; question: but why make coeffs between 0 and 0.1 instead of -0.5 and 0.5


In order to do matrix-matrix products (which we'll need in the next section), we need to turn `coeffs` into a column vector (i.e. a matrix with a single column), which we can do by passing a second argument `1` to `torch.rand()`, indicating that we want our coefficients to have one column:

```python
def init_coeffs(): return (torch.rand(n_coeff, 1)*0.1).requires_grad_()
```

### how to turn a single column of dependent variable into a single column matrix or a column vector


We'll also need to turn our dependent variable into a column vector, which we can do by indexing the column dimension with the special value `None`, which tells PyTorch to add a new dimension in this position:

```python
trn_dep = trn_dep[:,None]
val_dep = val_dep[:,None]
```

### question: why set learning rate to be 100 for this Titanic model


We can now train our model as before and confirm we get identical outputs...:

```python
coeffs = train_model(lr=100)
```

...and identical accuracy:

```python
acc(coeffs)
```

## A neural network


### how to initialize coeffs for a neuralnet with two layers (including a hidden layer of n neurons) and the final output layer is a single neuron with a single coeff; question: how do `-0.5` and `-0.3` come from?


We've now got what we need to implement our neural network.

First, we'll need to create coefficients for each of our layers. Our first set of coefficients will take our `n_coeff` inputs, and create `n_hidden` outputs. We can choose whatever `n_hidden` we like -- a higher number gives our network more flexibility, but makes it slower and harder to train. So we need a matrix of size `n_coeff` by `n_hidden`. We'll divide these coefficients by `n_hidden` so that when we sum them up in the next layer we'll end up with similar magnitude numbers to what we started with.

Then our second layer will need to take the `n_hidden` inputs and create a single output, so that means we need a `n_hidden` by `1` matrix there. The second layer will also need a constant term added.

```python
def init_coeffs(n_hidden=20):
    layer1 = (torch.rand(n_coeff, n_hidden)-0.5)/n_hidden
    layer2 = torch.rand(n_hidden, 1)-0.3
    const = torch.rand(1)[0]
    return layer1.requires_grad_(),layer2.requires_grad_(),const.requires_grad_()
```

### how to update `calc_preds` for this 2 layer neuralnet using `F.relu`, matrix product `@`, and `torch.sigmoid`


Now we have our coefficients, we can create our neural net. The key steps are the two matrix products, `indeps@l1` and `res@l2` (where `res` is the output of the first layer). The first layer output is passed to `F.relu` (that's our non-linearity), and the second is passed to `torch.sigmoid` as before.

```python
import torch.nn.functional as F

def calc_preds(coeffs, indeps):
    l1,l2,const = coeffs
    res = F.relu(indeps@l1)
    res = res@l2 + const
    return torch.sigmoid(res)
```

### how to update coeffs layer by layer with `layer.sub_` and `layer.grad.zero_`


Finally, now that we have more than one set of coefficients, we need to add a loop to update each one:

```python
def update_coeffs(coeffs, lr):
    for layer in coeffs:
        layer.sub_(layer.grad * lr)
        layer.grad.zero_()
```

### question: how the learning rate is chosen (1.4 or 20) when training


That's it -- we're now ready to train our model!

```python
coeffs = train_model(lr=1.4)
```

```python
coeffs = train_model(lr=20)
```

It's looking good -- our loss is lower than before. Let's see if that translates to a better result on the validation set:

```python
acc(coeffs)
```

In this case our neural net isn't showing better results than the linear model. That's not surprising; this dataset is very small and very simple, and isn't the kind of thing we'd expect to see neural networks excel at. Furthermore, our validation set is too small to reliably see much accuracy difference. But the key thing is that we now know exactly what a real neural net looks like!


## Deep learning


### how to move from neuralnet with one hidden layer to a deep learning


The neural net in the previous section only uses one hidden layer, so it doesn't count as "deep" learning. But we can use the exact same technique to make our neural net deep, by adding more matrix multiplications.

First, we'll need to create additional coefficients for each layer:

```python
def init_coeffs():
    hiddens = [10, 10]  # <-- set this to the size of each hidden layer you want
    sizes = [n_coeff] + hiddens + [1]
    n = len(sizes)
    layers = [(torch.rand(sizes[i], sizes[i+1])-0.3)/sizes[i+1]*4 for i in range(n-1)]
    consts = [(torch.rand(1)[0]-0.5)*0.1 for i in range(n-1)]
    for l in layers+consts: l.requires_grad_()
    return layers,consts
```

### why so many messy constants and how they block the progress of deep learning in the early days


You'll notice here that there's a lot of messy constants to get the random numbers in just the right ranges. When you train the model in a moment, you'll see that the tiniest changes to these initialisations can cause our model to fail to train at all! This is a key reason that deep learning failed to make much progress in the early days -- it's very finicky to get a good starting point for our coefficients. Nowadays, we have ways to deal with that, which we'll learn about in other notebooks.

Our deep learning `calc_preds` looks much the same as before, but now we loop through each layer, instead of listing them separately:


### how to use `enumerate` to loop both idx and item

```python
import torch.nn.functional as F

def calc_preds(coeffs, indeps):
    layers,consts = coeffs
    n = len(layers)
    res = indeps
    for i,l in enumerate(layers):
        res = res@l + consts[i]
        if i!=n-1: res = F.relu(res)
    return torch.sigmoid(res)
```

We also need a minor update to `update_coeffs` since we've got `layers` and `consts` separated now:

```python
def update_coeffs(coeffs, lr):
    layers,consts = coeffs
    for layer in layers+consts:
        layer.sub_(layer.grad * lr)
        layer.grad.zero_()
```

Let's train our model...

```python
coeffs = train_model(lr=4)
```

...and check its accuracy:

```python
acc(coeffs)
```

## Final thoughts


### How much similar or different between practical models and the models from scratch above


It's actually pretty cool that we've managed to create a real deep learning model from scratch and trained it to get over 80% accuracy on this task, all in the course of a single notebook!

The "real" deep learning models that are used in research and industry look very similar to this, and in fact if you look inside the source code of any deep learning model you'll recognise the basic steps are the same.

The biggest differences in practical models to what we have above are:

- How initialisation and normalisation is done to ensure the model trains correctly every time
- Regularization (to avoid over-fitting)
- Modifying the neural net itself to take advantage of knowledge of the problem domain
- Doing gradient descent steps on smaller batches, rather than the whole dataset.

I'll be adding notebooks about all these later, and will add links here once they're ready.

If you found this notebook useful, please remember to click the little up-arrow at the top to upvote it, since I like to know when people have found my work useful, and it helps others find it too. (BTW, be sure you're looking at my [original notebook here](https://www.kaggle.com/code/jhoward/linear-model-and-neural-net-from-scratch) when you do that, and are not on your own copy of it, otherwise your upvote won't get counted!) And if you have any questions or comments, please pop them below -- I read every comment I receive!

```python
from fastdebug.utils import *
```

```python
fastnbs("numpy book")
```

```python
fastlistnbs()
```

```python
ipy2md()
```
