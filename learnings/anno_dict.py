
def foo(a, b): pass
test_eq(foo.__annotations__, {})
test_eq(anno_dict(foo), {})

from fastcore.foundation import L
def _f(a:int, b:L)->str: ...
test_eq(_f.__annotations__, {'a': int, 'b': L, 'return': str})
test_eq(anno_dict(_f), {'a': int, 'b': L, 'return': str})

def anno_dict(f):=========================================================================(0)       
    "`__annotation__ dictionary with `empty` cast to `None`, returning empty if doesn't exist" # [34;1mempty2none works on paramter.default especially when the default is Parameter.empty[0m; [92;1manno_dict works on the types of params, not the value of params[0m; [37;1mso it is odd to use empty2none in anno_dict;[0m;  (1)
    return {k:empty2none(v) for k,v in getattr(f, '__annotations__', {}).items()}=========(2)       
                                                                                                                                                        (3)
