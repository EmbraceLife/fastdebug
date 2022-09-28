class BypassNewMeta(FixSigMeta):==========================================================(0)       
    "Metaclass: casts `x` to this class if it's of type `cls._bypass_type`"===============(1) # [34;1mBypassNewMeta allows its instance class e.g., _T to choose a specific class e.g., _TestB and change `__class__` of an object e.g., t of _TestB to _T without creating a new object[0m; 
    def __call__(cls, x=None, *args, **kwargs):===========================================(2)       
        if hasattr(cls, '_new_meta'): x = cls._new_meta(x, *args, **kwargs)===============(3) # [92;1mIf the instance class like _T has attr '_new_meta', then run it with param x;[0m; 
        elif not isinstance(x,getattr(cls,'_bypass_type',object)) or len(args) or len(kwargs): # [93;1mwhen x is not an instance of _T's _bypass_type[0m; [91;1mor when a positional param is given[0m; [36;1mor when a keyword arg is given[0m; [35;1mlet's run _T's super's __call__ function with x as param[0m; [92;1mand assign the result to x[0m;  (4)
            x = super().__call__(*((x,)+args), **kwargs)==================================(5)       
        if cls!=x.__class__: x.__class__ = cls============================================(6) # [37;1mIf x.__class__ is not cls or _T, then make it so[0m; 
        return x==========================================================================(7)       
                                                                                                                                                        (8)
