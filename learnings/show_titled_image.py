
%%snoop
show_titled_image((im3,'A puppy'), figsize=(2,2))

@snoop====================================================================================(0)       
@delegates(show_image, keep=True)=========================================================(1)       
def show_titled_image(o, **kwargs):=======================================================(2)       
    "Call `show_image` destructuring `o` to `(img,title)`"================================(3) # [91;1muse a tuple o which is (img, title) to make show_image to display the image with title printed on top[0m; 
    show_image(o[0], title=str(o[1]), **kwargs)===========================================(4)       
                                                                                                                                                        (5)
