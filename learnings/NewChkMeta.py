
class _T(metaclass=NewChkMeta):
    "Testing with metaclass NewChkMeta"
    def __init__(self, o=None, b=1):
        # if `o` is not an object without an attribute `foo`, set foo = 1
        self.foo = getattr(o,'foo',1)
        self.b = b

t = _T(3)
test_eq(t.foo,1) # 1 was not of type _T, so foo = 1

t2 = _T(t) # t2 will now reference t

test_is(t, t2) # t and t2 are the same object
t2.foo = 5 # this will also change t.foo to 5 because it is the same object
test_eq(t.foo, 5)
test_eq(t2.foo, 5)

t3 = _T(t, b=1)
assert t3 is not t

t4 = _T(t) # without any arguments the constructor will return a reference to the same object
assert t4 is t

class NewChkMeta(FixSigMeta):=============================================================(0)       
    "Metaclass to avoid recreating object passed to constructor"==========================(1) # [93;1mNewChkMeta is a metaclass inherited from FixSigMea[0m; [34;1mit makes its own __call__[0m; [93;1mwhen its class instance, e.g., _T, create object instances (e.g, t) without args nor kwargs but only x, and x is an object of the instance class, then return x[0m; [34;1motherwise, create and return a new object created by the instance class's super class' __call__ method with x as param[0m; [93;1mIn other words, t = _T(3) will create a new obj[0m; [34;1m_T(t) will return t[0m; [92;1m_T(t, 1) or _T(t, b=1) will also return a new obj[0m; 
    def __call__(cls, x=None, *args, **kwargs):===========================================(2) # [93;1mhow to create a __call__ method with param cls, x, *args, **kwargs;[0m; 
        if not args and not kwargs and x is not None and isinstance(x,cls): return x======(3) # [34;1mhow to express no args and no kwargs and x is an instance of cls?[0m; 
        res = super().__call__(*((x,) + args), **kwargs)==================================(4) # [36;1mhow to call __call__ of super class with x and consider all possible situations of args and kwargs[0m; 
        return res========================================================================(5)       
                                                                                                                                                        (6)
