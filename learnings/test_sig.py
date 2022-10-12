
def func_2(h,i=3, j=[5,6]): pass
test_sig(func_2, '(h, i=3, j=[5, 6])')

def test_sig(f, b):=======================================================================(0)       
    "Test the signature of an object"=====================================================(1) # [36;1mtest_sig(f:FunctionType or ClassType, b:str)[0m; [91;1mtest_sig will get f's signature as a string[0m; [92;1mb is a signature in string provided by the user[0m; [91;1min fact, test_sig is to compare two strings[0m; 
    test_eq(str(inspect.signature(f)), b)=================================================(2) # [35;1mtest_sig is to test two strings with test_eq[0m; [36;1mhow to turn a signature into a string;[0m; 
                                                                                                                                                        (3)
