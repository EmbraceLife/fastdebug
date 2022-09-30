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

# 0010_fastcore_meta_summary


## import

```python
from fastdebug.utils import *
from fastdebug.core import *
```

## fastcore and fastcore.meta

```python
import fastcore
```

```python
whichversion("fastcore")
```

```python
whatinside(fastcore, lib=True)
```

```python
from fastcore.meta import *
import fastcore.meta as fm
```

### What's inside fastcore.meta

```python
whatinside(fm, dun=True)
```

## Review individual funcs and classes


### What is fastcore.meta all about? 

It is a submodule contains 4 metaclasses, 1 class built by a metaclass, 4 decorators and a few functions.    

Metaclasses give us the power to create new breeds of classes with new features.     

Decorators give us the power to add new features to existing funcions.    

We can find their basic info [above](#What's-inside-fastcore.meta)


### What can these metaclasses do for me?

We design/create classes to breed objects as we like.

We design/create metaclasses to breed classes as we like.

Before metaclasses, all classes are created by type and are born the same.

With metaclasses, e.g., FixSigMeta first uses type to its instance classes exactly like above, but then FixSigMeta immediately adds new features to them right before they are born.


#### FixSigMeta
can breed classes which are free of signature problems (or they can automatically fix signature problems).


#### PrePostInitMeta
inherited/evolved from `FixSigMeta` to breed classes which can initialize their objects using `__pre_init__`, 
`__init__`, `__post_init__` whichever is available (allow me to abbreviate it as triple_init).


#### AutoInit
is an instance class created by `PrePostInitMeta`, and together with its own defined `__pre_init__`, subclasses of `AutoInit` has to worry about running `super().__init__(...)` no more.


- As `AutoInit` is an instance class created by `PrePostInitMeta`, it can pass on both features (free of signature problem and triple_init) to its subclasses. 
- As it also defines its own `__pre_init__` function which calls its superclass `__init__` function, its subclasses will inherit this `__pre_init__` function too.
- When subclasses of `AutoInit` create and initialize object intances through `__call__` from `PrePostInitMeta`, `AutoInit`'s `__pre_init__` runs `super().__init__(...)`, so when we write `__init__` function of subclasses which inherits from `AutoInit`, we don't need to write `super().__init__(...)` any more.


#### NewChkMeta


is inherited from `FixSigMeta`, so any instance classes created by `NewChkMeta` can also pass on the no_signature_problem feature.

It defines its own `__call__` to enable all the instance objects e.g., `t` created by all the instance classes e.g., `T` created by `NewChkMeta` to do the following: 

- `T(t) is t if isinstance(t, T)` returns true
- when `T(t) is t if not isinstance(t, T)`, or when `T(t, 1) is t if isinstance(t, T)` or when `T(t, b=1) is t if isinstance(t, T)`, all return False

In other words, `NewChkMeta` creates a new breed of classes `T` as an example which won't recreate the same instance object `t` twice. But if `t` is not `T`'s instance object, or we choose to add more flavor to `t`, then `T(t)` or `T(t, 1)` will create a new instance object of `T`.


#### BypassNewMeta


is inherited from `FixSigMeta`, so it has the feature of free from signature problems.


It defines its own `__call__`, so that when its instance classes `_T` create and initialize objects with a param `t` which is an instance object of another class `_TestB`, they can do the following:


- If `_T` likes `_TestB` and prefers `t` as it is, then when we run `t2 = _T(t)`, and `t2 is t` will be True, and both are instances of `_T`.  
- If `_T` is not please with `t`, it could be that `_T` does not like `_TestB` any more, then `_T(t) is t` will be False
- or maybe `_T` still likes `_TestB`, but want to add some flavors to `t` by `_T(t, 1)` or `_T(t, b=1)` for example, in this case `_T(t) is t` will also be False.


In other words, `BypassNewMeta` creates a new breed of instance classes `_T` which don't need to create but make an object `t` of its own object instance, if `t` is an instance object of `_TestB` which is liked by `_T` and if `_T` likes `t` as it is.


### What can those decorators do for me?

A decorator is a function that takes in a function and returns a modified function.

A decorator allows us to modify the behavior of a function. 


#### use_kwargs_dict


allows us to replace an existing function's param `kwargs` with a number of params with default values.

The params with their default values are provided in a dictionary.


#### use_kwargs


allows us to replace an existing function's param `kwargs` with a number of params with None as their default values.

The params are provided as names in a list.


#### delegates


allows us to replace an existing function's param `kwargs` with a number of params with their default values from another existing function.

In fact, `delegates` can work on function, classes, and methods.


#### funcs_kwargs
is a decorator to classes. It can help classes to bring in existing functions as their methods. 

It can set the methods to use or not use `self` in the class.


### The remaining functions


`test_sig` and `method` are straightforward, their docs tell it all clearly.


`empty2none` and `anno_dict` are no in use much at all. see the thread [here](). 

```python
fastview("FixSigMeta", nb=True)
```

```python
%debug
```

```python
fastsrcs()
```

```python
fastview(FixSigMeta)
```

```python
fastcodes("how to get signature's parameters", accu=0.8, nb=True, db=True)
```

```python
fastnotes("how make the most", folder="all")
```

## What is fastcore.meta all about

```python

```

```python

```

```python

```

```python

```
