
_,axs = subplots()
test_eq(axs.shape,[1])
plt.close()

@snoop====================================================================================(0)       
@delegates(plt.subplots, keep=True)=======================================================(1)       
def subplots(=============================================================================(2)       
    nrows:int=1, # Number of rows in returned axes grid===================================(3)       
    ncols:int=1, # Number of columns in returned axes grid================================(4)       
    figsize:tuple=None, # Width, height in inches of the returned figure =================(5)       
    imsize:int=3, # Size (in inches) of images that will be displayed in the returned figure                                                            (6)
    suptitle:str=None, # Title to be set to returned figure===============================(7)       
    **kwargs==============================================================================(8)       
) -> (plt.Figure, plt.Axes): # Returns both fig and ax as a tuple ========================(9)       
    "Returns a figure and set of subplots to display images of `imsize` inches"===========(10) # [37;1mA wrapper around plt.subplots(...)[0m; [92;1monly used in show_images[0m; [91;1mnot in show_image, nor in show_image_batch;to create/display and return a fig with specified size and a specified num of empty subplots[0m; 
    if figsize is None: ==================================================================(11)      
        pp.deep(lambda: nrows*imsize if suptitle is None or imsize>2 else nrows*imsize+0.6)                                                             (12)
        h=nrows*imsize if suptitle is None or imsize>2 else nrows*imsize+0.6 #https://github.com/matplotlib/matplotlib/issues/5355                      (13)
        figsize=(ncols*imsize, h)=========================================================(14)      
    fig,ax = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)========================(15)      
    if suptitle is not None: fig.suptitle(suptitle)=======================================(16)      
    if nrows*ncols==1: ax = array([ax])===================================================(17)      
    return fig,ax=========================================================================(18)      
                                                                                                                                                        (19)
