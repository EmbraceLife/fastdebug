
class Foo(metaclass=FixSigMeta):
    def __init__(self): pass

class FixSigMeta(type):===================================================================(0)       
    "A metaclass that fixes the signature on classes that override `__new__`"=============(1) # [91;1mFixSigMeta inherits __init__, and __call__ from type[0m; [91;1mbut writes its own __new__[0m; [93;1mFoo inherits all three from type[0m; [91;1mFixSigMeta is used to create class instance not object instance.[0m; 
    def __new__(cls, name, bases, dict):==================================================(2)       
        res = super().__new__(cls, name, bases, dict)=====================================(3) # [36;1mhow to create a new class instance with type dynamically[0m; [92;1mthe rest below is how FixSigMeta as a metaclass create its own instance classes[0m; 
        if res.__init__ is not object.__init__: res.__signature__ = _rm_self(inspect.signature(res.__init__)) # [92;1mhow to check whether a class has its own __init__ function[0m; [93;1mhow to remove self param from a signature[0m;  (4)
        return res========================================================================(5)       
                                                                                                                                                        (6)
