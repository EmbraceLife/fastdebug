def _rm_self(sig):========================================================================(0) # [93;1mremove parameter self from a signature which has self;[0m; 
    sigd = dict(sig.parameters)===========================================================(1) # [35;1mhow to access parameters from a signature[0m; [35;1mhow is parameters stored in sig[0m; [34;1mhow to turn parameters into a dict;[0m; 
    sigd.pop('self')======================================================================(2) # [36;1mhow to remove the self parameter from the dict of sig;[0m; 
    return sig.replace(parameters=sigd.values())==========================================(3) # [36;1mhow to update a sig using a updated dict of sig's parameters[0m; 
                                                                                                                                                        (4)
