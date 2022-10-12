
def low(a, b:int=1): pass
@delegates(low)
def mid(c, d:list=None, **kwargs): pass
pprint(inspect.signature(mid)) # pprint and inspect is loaded from fastdebug

def delegates(to:FunctionType=None, # Delegatee===========================================(0) # [36;1mhow to make delegates(to) to have to as FunctionType and default as None[0m; 
              keep=False, # Keep `kwargs` in decorated function?==========================(1)       
              but:list=None): # Exclude these parameters from signature===================(2) # [93;1mhow to make delegates(to, but) to have 'but' as list and default as None[0m; 
    "Decorator: replace `**kwargs` in signature with params from `to`"====================(3)       
    if but is None: but = []==============================================================(4)       
    def _f(f):============================================================================(5)       
        if to is None: to_f,from_f = f.__base__.__init__,f.__init__=======================(6) # [93;1mhow to write 2 ifs and elses in 2 lines[0m; 
        else:          to_f,from_f = to.__init__ if isinstance(to,type) else to,f=========(7) # [93;1mhow to assign a,b together with if and else[0m; 
        from_f = getattr(from_f,'__func__',from_f)========================================(8) # [92;1mIs classmethod callable[0m; [92;1mdoes classmethod has __func__[0m; [37;1mcan we do inspect.signature(clsmethod)[0m; [92;1mhow to use getattr(obj, attr, default)[0m; 
        to_f = getattr(to_f,'__func__',to_f)==============================================(9)       
        if hasattr(from_f,'__delwrap__'): return f========================================(10) # [91;1mif B has __delwrap__, can we do delegates(A)(B) again?[0m; [34;1mhasattr(obj, '__delwrap__')[0m; 
        sig = inspect.signature(from_f)===================================================(11) # [91;1mhow to get signature obj of B[0m; [37;1mwhat does a signature look like[0m; [92;1mwhat is the type[0m; 
        sigd = dict(sig.parameters)=======================================================(12) # [91;1mHow to access parameters of a signature?[0m; [34;1mHow to turn parameters into a dict?[0m; 
        k = sigd.pop('kwargs')============================================================(13) # [36;1mHow to remove an item from a dict?[0m; [93;1mHow to get the removed item from a dict?[0m; [92;1mHow to add the removed item back to the dict?[0m; [37;1mwhen writing expressions, as they share environment, so they may affect the following code[0m; 
        s2 = {k:v.replace(kind=inspect.Parameter.KEYWORD_ONLY) for k,v in inspect.signature(to_f).parameters.items() # [34;1mHow to access a signature's parameters as a dict?[0m; [93;1mHow to replace the kind of a parameter with a different kind?[0m; [92;1mhow to check whether a parameter has a default value?[0m; [34;1mHow to check whether a string is in a dict and a list?[0m; [34;1mhow dict.items() and dict.values() differ[0m;  (14)
              if v.default != inspect.Parameter.empty and k not in sigd and k not in but}=(15)      
        anno = {k:v for k,v in getattr(to_f, "__annotations__", {}).items() if k not in sigd and k not in but} # [91;1mHow to get A's __annotations__?[0m; [91;1mHow to access it as a dict?[0m; [91;1mHow to select annotations of the right params with names?[0m; [35;1mHow to put them into a dict?[0m; [92;1mHow to do it all in a single line[0m;  (16)
        sigd.update(s2)===================================================================(17) # [34;1mHow to add the selected params from A's signature to B's signature[0m; [37;1mHow to add items into a dict;[0m; 
        if keep: sigd['kwargs'] = k=======================================================(18) # [37;1mHow to add a new item into a dict;[0m; 
        else: from_f.__delwrap__ = to_f===================================================(19) # [93;1mHow to create a new attr for a function or obj;[0m; 
        from_f.__signature__ = sig.replace(parameters=sigd.values())======================(20) # [36;1mHow to update a signature with a new set of parameters;[0m; 
        if hasattr(from_f, '__annotations__'): from_f.__annotations__.update(anno)========(21) # [36;1mHow to check whether a func has __annotations__[0m; [34;1mHow add selected params' annotations from A to B's annotations;[0m; 
        return f==========================================================================(22)      
    return _f=============================================================================(23)      
                                                                                                                                                        (24)
