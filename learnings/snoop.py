
@patch====================================================================================(0)       
def snoop(self:Fastdb, watch:list=None, deco=False, db=False):============================(1)       
    "run snoop on the func or class under investigation only when example is available"===(2)       
    self.idxsrc = None # so that autoprint won't print src at all for snoop===============(3)       
    self.printtitle() # maybe at some point, I should use await to make the output of printtitle appear first                                           (4)
    if bool(self.eg):=====================================================================(5)       
        self.create_snoop_str(watch=watch, deco=deco, db=db)==============================(6)       
        self.create_snoop_from_string(db=db)==============================================(7)       
        self.run_example()================================================================(8)       
                                                                                                                                                        (9)
