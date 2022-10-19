
%%snoop
show_images((im,im3),titles=('number','puppy'),suptitle='Number Puppy',  imsize=3)

@delegates(subplots)======================================================================(0)       
def show_images(ims, nrows=1, ncols=None, titles=None, **kwargs):=========================(1)       
    "Show all images `ims` as subplots with `rows` using `titles`."=======================(2)       
    if ncols is None: ncols = int(math.ceil(len(ims)/nrows))==============================(3) # [91;1mto display multiple images with/wihtout titles in rows and cols[0m; [34;1mfirst input 'ims' is a tuple of imgs[0m; [34;1m'titles' should be in tuple if not None[0m; [35;1mncols is calc by total imgs and nrows;titles are given as None for all images if not available[0m; [93;1mempty fig and subplots are created by subplots[0m; [92;1mfinally loop through each img, title, ax to show_image[0m; [93;1m[0m; 
    if titles is None: titles = [None]*len(ims)===========================================(4)       
#     pp.deep(lambda: subplots(nrows, ncols, **kwargs)[1].flat)===========================(5)       
    axs = subplots(nrows, ncols, **kwargs)[1].flat========================================(6)       
    for im,t,ax in zip(ims, titles, axs): show_image(im, ax=ax, title=t)==================(7)       
                                                                                                                                                        (8)
