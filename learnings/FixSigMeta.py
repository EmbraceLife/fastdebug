class FixSigMeta(type):===================================================================(0)       
    "A metaclass that fixes the signature on classes that override `__new__`"=============(1) # [36;1mAny class having FixSigMeta as metaclass will have its own __init__ func stored in its attr __signature__;FixSigMeta uses its __new__ to create a class instance[0m; [34;1mthen check whether its class instance has its own __init__;if so, remove self from the sig of __init__[0m; [92;1mthen assign this new sig to __signature__ for the class instance;[0m; 
    def __new__(cls, name, bases, dict):==================================================(2) # [36;1mhow does a metaclass create a class instance[0m; [93;1mwhat does super().__new__() do here;[0m; 
        res = super().__new__(cls, name, bases, dict)=====================================(3)       
        if res.__init__ is not object.__init__: res.__signature__ = _rm_self(inspect.signature(res.__init__)) # [35;1mhow to remove self from a signature[0m; [92;1mhow to check whether a class' __init__ is inherited from object or not;[0m;  (4)
        return res========================================================================(5)       
                                                                                                                                                        (6)
