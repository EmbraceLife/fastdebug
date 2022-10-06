
class TestParent():
    def __init__(self): self.h = 10
        
class TestChild(AutoInit, TestParent):
    def __init__(self): self.k = self.h + 2
    
t = TestChild()
test_eq(t.h, 10) # h=10 is initialized in the parent class
test_eq(t.k, 12)

class AutoInit(metaclass=PrePostInitMeta):================================================(0)       
    "Same as `object`, but no need for subclasses to call `super().__init__`"=============(1) # [34;1mAutoInit inherit __new__ and __init__ from object to create and initialize object instances[0m; [93;1mAutoInit uses PrePostInitMeta.__new__ or in fact FixSigMeta.__new__ to create its own class instance, which can have __signature__[0m; [34;1mAutoInit uses PrePostInitMeta.__call__ to specify how its object instance to be created and initialized (with pre_init, init, post_init))[0m; [91;1mAutoInit as a normal or non-metaclass, it writes its own __pre_init__ method[0m; 
    def __pre_init__(self, *args, **kwargs): super().__init__(*args, **kwargs)============(2) # [34;1mhow to run superclass' __init__ function[0m; 
                                                                                                                                                        (3)
