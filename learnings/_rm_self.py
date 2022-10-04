
class Foo:
    def __init__(self, a, b:int=1): pass
pprint(inspect.signature(Foo.__init__))
pprint(_rm_self(inspect.signature(Foo.__init__)))

def _rm_self(sig):========================================================================(0) # [34;1mremove parameter self from a signature which has self;[0m; 
    sigd = dict(sig.parameters)===========================================================(1) # [34;1mhow to access parameters from a signature[0m; [92;1mhow is parameters stored in sig[0m; [92;1mhow to turn parameters into a dict;[0m; 
    sigd.pop('self')======================================================================(2) # [34;1mhow to remove the self parameter from the dict of sig;[0m; 
    return sig.replace(parameters=sigd.values())==========================================(3) # [92;1mhow to update a sig using a updated dict of sig's parameters[0m; 
                                                                                                                                                        (4)
