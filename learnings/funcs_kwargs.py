
@funcs_kwargs
class T:
    _methods=['b'] # allows you to add method b upon instantiation
    def __init__(self, f=1, **kwargs): pass # don't forget to include **kwargs in __init__
    def a(self): return 1
    def b(self): return 2
    
t = T()
test_eq(t.a(), 1)
test_eq(t.b(), 2)

test_sig(T, '(f=1, *, b=None)')
inspect.signature(T)

def _new_func(): return 5

t = T(b = _new_func)
test_eq(t.b(), 5)

t = T(a = lambda:3)
test_eq(t.a(), 1) # the attempt to add a is ignored and uses the original method instead.

def _f(self,a=1): return self.num + a # access the num attribute from the instance

@funcs_kwargs(as_method=True)
class T: 
    _methods=['b']
    num = 5
    
t = T(b = _f) # adds method b
test_eq(t.b(5), 10) # self.num + 5 = 10

def _f(self,a=1): return self.num * a #multiply instead of add 

class T2(T):
    def __init__(self,num):
        super().__init__(b = _f) # add method b from the super class
        self.num=num
        
t = T2(num=3)
test_eq(t.b(a=5), 15) # 3 * 5 = 15
test_sig(T2, '(num)')

def funcs_kwargs(as_method=False):========================================================(0)       
    "Replace methods in `cls._methods` with those from `kwargs`"==========================(1) # [34;1mhow funcs_kwargs works[0m; [93;1mit is a wrapper around _funcs_kwargs[0m; [91;1mit offers two ways of running _funcs_kwargs[0m; [36;1mthe first, default way, is to add a func to a class without using self[0m; [93;1msecond way is to add func to class enabling self use;[0m; 
    if callable(as_method): return _funcs_kwargs(as_method, False)========================(2) # [91;1mhow to check whether an object is callable[0m; [36;1mhow to return a result of running a func[0m; [35;1m[0m; 
    return partial(_funcs_kwargs, as_method=as_method)====================================(3) # [36;1mhow to custom the params of `_funcs_kwargs` for a particular use with partial[0m; 
                                                                                                                                                        (4)
