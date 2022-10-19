
def _fig_bounds(x):=======================================================================(0) # [92;1mmake sure figsize is properly defined[0m; [35;1mthe bounds (either width or height) is between 1 and 5[0m; 
    "make sure figsize is properly defined; the bounds (either width or height) is between 1 and 5"                                                     (1)
#     pp.deep(lambda: x//32)==============================================================(2)       
    r = x//32=============================================================================(3)       
#     pp.deep(lambda: min(5, max(1,r)))===================================================(4)       
    return min(5, max(1,r))===============================================================(5) # [37;1monly allow value from 1 to 5[0m; 
                                                                                                                                                        (6)
