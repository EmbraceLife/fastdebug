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

# Explore and Document Fastcore.meta.delegates

```python
# from IPython.core.display import display, HTML # a depreciated import
from IPython.display import display, HTML 

display(HTML("<style>.container { width:100% !important; }</style>"))
```

## Import

```python
from fastdebug.core import *
```

```python
from fastcore.meta import delegates 
```

## Initiate Fastdb and example in str

```python
g = locals() # this is a must
fdb = Fastdb(delegates, outloc=g)
```

## Example

```python
def low(a, b=1): pass
@delegates(low) # this format is fine too
def mid(c, d=1, **kwargs): pass
pprint(inspect.signature(mid)) # pprint and inspect is loaded from fastdebug
```

```python
def low(a, b=1): pass
def mid(c, d=1, **kwargs): pass
pprint(inspect.signature(delegates(low)(mid))) 
```

```python
fdb.eg = """
def low(a, b=1): pass
def mid(c, d=1, **kwargs): pass
pprint(inspect.signature(delegates(low)(mid)))
"""
```

```python
fdb.eg = """
def low(a, b=1): pass
@delegates(low)
def mid(c, d=1, **kwargs): pass
pprint(inspect.signature(mid)) # pprint and inspect is loaded from fastdebug
"""
```

```python
fdb.eg = """
class Foo:
    @classmethod
    def clsm(a, b=1): pass
    
    @delegates(clsm)
    def instm(c, d=1, **kwargs): pass
f = Foo()
pprint(inspect.signature(f.instm)) # pprint and inspect is loaded from fastdebug
"""
```

```python
fdb.eg = """
def low(a, b:int=1): pass
@delegates(low)
def mid(c, d:list=None, **kwargs): pass
pprint(inspect.signature(mid)) # pprint and inspect is loaded from fastdebug
"""
```

## docsrc

```python
fdb.eg
```

```python
fdb.print()
```

```python
fdb.docsrc(21, "How to check whether a func has __annotations__; How add selected params' annotations from A to B's annotations;")

fdb.docsrc(14, "How to access a signature's parameters as a dict?; How to replace the kind of a parameter with a different kind?; \
how to check whether a parameter has a default value?; How to check whether a string is in a dict and a list?; \
how dict.items() and dict.values() differ", \
"inspect.signature(to_f).parameters.items()", "inspect.signature(to_f).parameters.values()", "inspect.Parameter.empty", \
"for k,v in inspect.signature(to_f).parameters.items():\\n\
    printright(f'v.kind: {v.kind}')\\n\
    printright(f'v.default: {v.default}')\\n\
    v1 = v.replace(kind=inspect.Parameter.KEYWORD_ONLY)\\n\
    printright(f'v1.kind: {v1.kind}')")

fdb.docsrc(16, "How to get A's __annotations__?; How to access it as a dict?; \
How to select annotations of the right params with names?; How to put them into a dict?; How to do it all in a single line", \
          "getattr(to_f, '__annotations__', {})", "getattr(from_f, '__annotations__', {})")
fdb.docsrc(17, "How to add the selected params from A's signature to B's signature; How to add items into a dict;")
fdb.docsrc(18, "How to add a new item into a dict;")
fdb.docsrc(19, "How to create a new attr for a function or obj;")
fdb.docsrc(20, "How to update a signature with a new set of parameters;", "sigd", "sigd.values()", "sig", \
           "sig.replace(parameters=sigd.values())")


fdb.docsrc(2, "how to make delegates(to, but) to have 'but' as list and default as None")
fdb.docsrc(0, "how to make delegates(to) to have to as FunctionType and default as None")
fdb.docsrc(6, "how to write 2 ifs and elses in 2 lines")
fdb.docsrc(7, "how to assign a,b together with if and else")
fdb.docsrc(8, "Is classmethod callable; does classmethod has __func__; can we do inspect.signature(clsmethod); \
how to use getattr(obj, attr, default)", \
           "from_f", "to_f", \
           "hasattr(from_f, '__func__')", "from_f = getattr(from_f,'__func__',from_f)", \
           "hasattr(to_f, '__func__')", "from_f = getattr(to_f,'__func__',to_f)", "callable(to_f)")

fdb.docsrc(10, "if B has __delwrap__, can we do delegates(A)(B) again?; hasattr(obj, '__delwrap__')")
fdb.docsrc(11, "how to get signature obj of B; what does a signature look like; what is the type", "inspect.signature(from_f)", "type(inspect.signature(from_f))")
fdb.docsrc(12, "How to access parameters of a signature?; How to turn parameters into a dict?", "sig.parameters", "dict(sig.parameters)")
fdb.docsrc(13, "How to remove an item from a dict?; How to get the removed item from a dict?; How to add the removed item back to the dict?; \
when writing expressions, as they share environment, so they may affect the following code", "sigd", "k = sigd.pop('kwargs')", "sigd", "sigd['kwargs'] = k")


# fdb.docsrc(3,"Personal Docs: delegates(A)(B), given B has kwargs in sig; delegating params of A to B; A and B can be funcs, methods, classes; \
# when A or B are classmethod or if they have __func__, then __func__ shall be used to get sig/params; \
# when A and B are classes, A.__init__ is actually used to get sig/params, B offers sig/params as a class; \
# B only take wanted and missing params with default values; B can keep its kwargs, if not, B will store A in __delwrap__ to prevent delegates(A)(B) again;\
# B will get __signature__ and store its new sig; if B has __annotations__, then add what B want from A's __annotations__")
```

```python
fdb.print()
```

## Snoop

```python
# fdb.snoop(deco=True) # both examples above works for Fastdb
```

```python
#| hide
!jupytext --to md /Users/Natsume/Documents/fastdebug/Demos/fastcore_meta_delegates.ipynb
!mv /Users/Natsume/Documents/fastdebug/Demos/fastcore_meta_delegates.md \
/Users/Natsume/Documents/divefastai/Debuggable/jupytext/

!jupyter nbconvert --config /Users/Natsume/Documents/mynbcfg.py --to markdown \
--output-dir /Users/Natsume/Documents/divefastai/Debuggable/nbconvert
```

```python
'delegates' in fdb.outenv
```

```python
delegates??
```

```python
fdb.dbsrc??
```

```python

```
