
def a(x=2): return x + 1
assert type(a).__name__ == 'function' # how to test on the type of function or method

a = method(a)
assert type(a).__name__ == 'method'

def method(f):============================================================================(0)       
    "Mark `f` as a method"================================================================(1)       
    # `1` is a dummy instance since Py3 doesn't allow `None` any more=====================(2)       
    return MethodType(f, 1)===============================================================(3)       
                                                                                                                                                        (4)
