
@funcs_kwargs
class T:
    _methods=['b'] # allows you to add method b upon instantiation
    def __init__(self, f=1, **kwargs): pass # don't forget to include **kwargs in __init__
    def a(self): return 1
    def b(self): return 2
    
t = T()
test_eq(t.a(), 1)
test_eq(t.b(), 2)

test_sig(T, '(f=1, *, b=None)')
inspect.signature(T)

def _new_func(): return 5

t = T(b = _new_func)
test_eq(t.b(), 5)

t = T(a = lambda:3)
test_eq(t.a(), 1) # the attempt to add a is ignored and uses the original method instead.


def _funcs_kwargs(cls, as_method):========================================================(0) # [93;1mhow does _funcs_kwargs work: _funcs_kwargs is a decorator[0m; [35;1mit helps class e.g., T to add more methods[0m; [34;1mI need to give the method a name, and put the name e.g., 'b' inside a list called _methods=['b'] inside class T[0m; [37;1mthen after writing a func e.g., _new_func, I can add it by T(b = _new_func)[0m; [93;1mif I want the func added to class to use self, I shall write @funcs_kwargs(as_method=True)[0m; 
    old_init = cls.__init__===============================================================(1)       
    def _init(self, *args, **kwargs):=====================================================(2) # [37;1mhow to define a method which can use self and accept any parameters[0m; 
        for k in cls._methods:============================================================(3) # [36;1mhow to pop out the value of an item in a dict (with None as default), and if the item name is not found, pop out None instead[0m; [92;1m[0m; 
            arg = kwargs.pop(k,None)======================================================(4)       
            if arg is not None:===========================================================(5)       
                if as_method: arg = method(arg)===========================================(6) # [34;1mhow to turn a func into a method[0m; 
                if isinstance(arg,MethodType): arg = MethodType(arg.__func__, self)=======(7) # [91;1mhow to give a method a different instance, like self[0m; 
                setattr(self, k, arg)=====================================================(8) # [36;1mhow to add a method to a class as an attribute[0m; 
        old_init(self, *args, **kwargs)===================================================(9)       
    functools.update_wrapper(_init, old_init)=============================================(10) # [34;1mhow to wrap `_init` around `old_init`, so that `_init` can use `old_init` inside itself[0m; 
    cls.__init__ = use_kwargs(cls._methods)(_init)========================================(11) # [37;1mhow to add a list of names with None as default value to function `_init` to repalce its kwargs param[0m; 
    if hasattr(cls, '__signature__'): cls.__signature__ = _rm_self(inspect.signature(cls.__init__)) # [36;1mhow to make a class.`__init__` signature to be the signature of the class using `__signature__` and `_rm_self`[0m;  (12)
    return cls============================================================================(13)      
                                                                                                                                                        (14)
