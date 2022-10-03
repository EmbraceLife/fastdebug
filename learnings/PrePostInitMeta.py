
class _T(metaclass=PrePostInitMeta):
    def __pre_init__(self):  self.a  = 0; 
    def __init__(self,b=0):  self.b = self.a + 1; assert self.b==1
    def __post_init__(self): self.c = self.b + 2; assert self.c==3

t = _T()
test_eq(t.a, 0) # set with __pre_init__
test_eq(t.b, 1) # set with __init__
test_eq(t.c, 3) # set with __post_init__
inspect.signature(_T)

class PrePostInitMeta(FixSigMeta):========================================================(0)       
    "A metaclass that calls optional `__pre_init__` and `__post_init__` methods"==========(1) # [36;1mPrePostInitMeta inherit __new__ and __init__ from FixSigMeta as a metaclass (a different type)[0m; [36;1mnot from type, nor from object[0m; [93;1mPrePostInitMeta is itself a metaclass, which is used to create class instance not object instance[0m; [91;1mPrePostInitMeta writes its own __call__ which regulates how its class instance create and initialize object instance[0m; 
    def __call__(cls, *args, **kwargs):===================================================(2)       
        res = cls.__new__(cls)============================================================(3) # [93;1mhow to create an object instance with a cls[0m; [91;1mhow to check the type of an object is cls[0m; [37;1mhow to run a function without knowing its params;[0m; 
        if type(res)==cls:================================================================(4)       
            if hasattr(res,'__pre_init__'): res.__pre_init__(*args,**kwargs)==============(5)       
            res.__init__(*args,**kwargs)==================================================(6) # [35;1mhow to run __init__ without knowing its params[0m; 
            if hasattr(res,'__post_init__'): res.__post_init__(*args,**kwargs)============(7)       
        return res========================================================================(8)       
                                                                                                                                                        (9)
