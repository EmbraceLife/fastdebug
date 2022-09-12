# AUTOGENERATED! DO NOT EDIT! File to edit: ../00_core.ipynb.

# %% auto 0
__all__ = ['defaults', 'dbcolors', 'colorize', 'strip_ansi', 'alignright', 'printsrclinewithidx', 'printsrc', 'dbprintinsert',
           'Fastdb', 'reliveonce']

# %% ../00_core.ipynb 9
defaults = type('defaults', (object,), {'margin': 157, # align to the right by 157
                                        'orisrc': None, # keep a copy of original official src code
                                        'outenv': globals(), # outside global env
                                        'cmts': {} # a dict to store idx and cmt
                                        # 'eg': None, # examples
                                        # 'src': None, # official src
                                       }) 

# %% ../00_core.ipynb 24
from pprint import pprint

# %% ../00_core.ipynb 35
import inspect

# %% ../00_core.ipynb 55
from fastcore.basics import *

# %% ../00_core.ipynb 56
class dbcolors:
    g = '\033[92m' #GREEN
    y = '\033[93m' #YELLOW
    r = '\033[91m' #RED
    reset = '\033[0m' #RESET COLOR

# %% ../00_core.ipynb 57
def colorize(cmt, color:str=None):
    if color == "g":
        return dbcolors.g + cmt + dbcolors.reset
    elif color == "y":
        return dbcolors.y + cmt + dbcolors.reset
    elif color == "r":
        return dbcolors.r + cmt + dbcolors.reset
    else: 
        return cmt

# %% ../00_core.ipynb 61
import re

# %% ../00_core.ipynb 62
def strip_ansi(source):
    return re.sub(r'\033\[(\d|;)+?m', '', source)

# %% ../00_core.ipynb 64
def alignright(blocks, margin:int=157):
    lst = blocks.split('\n')
    maxlen = max(map(lambda l : len(strip_ansi(l)) , lst ))
    indent = margin - maxlen
    for l in lst:
        print(' '*indent + format(l))

# %% ../00_core.ipynb 79
import inspect

# %% ../00_core.ipynb 94
def printsrclinewithidx(idx, l, fill=" "):
    totallen = 157
    lenidx = 5
    lenl = len(l)
    print(l + fill*(totallen-lenl-lenidx) + "(" + str(idx) + ")")

# %% ../00_core.ipynb 101
def printsrc(src, # name of src code such as foo, or delegates
             dbcode, # string of codes or int of code idx number
             cmt,
             expand:int=2): # expand the codes around the srcline under investigation
    "print the seleted srcline with comment, idx and specified num of expanding srclines"
    lstsrc = inspect.getsource(src).split('\n')
    
    dblines = ""
    if type(dbcode) == int:
        dblines = lstsrc[dbcode]
    else:
        dblines = dbcode
        numdblines = list(map(lambda x: bool(x.strip()), dblines.split('\n'))).count(True)
    
    
    dblineidx = []
    for idx, l in zip(range(len(lstsrc)), lstsrc):
        if bool(l) and l.strip() in dblines:
            dblineidx.append(idx)

    for idx, l in zip(range(len(lstsrc)), lstsrc):
        
        srcidx = dbcode if type(dbcode) == int else dblineidx[0]
        
        if bool(l) and l.strip() in dblines and idx == srcidx:
            printsrclinewithidx(idx, l, fill="=")

            if bool(cmt):
                colcmt = colorize(cmt, "r")
                alignright(colcmt) # also print the comment

        if idx >= srcidx - expand and idx < srcidx:
            printsrclinewithidx(idx, l)
        elif idx <= srcidx + expand and idx > srcidx:
            printsrclinewithidx(idx, l)


# %% ../00_core.ipynb 153
import ast

# %% ../00_core.ipynb 154
def dbprintinsert(*codes, env={}): 

        
    # trial and error version for real code, still not quite why globals vs locals work in exec and eval
    for c in codes:
        print("\n")
        
        # handle a block of code
        if "\n" in c: 
            output = f"Running your code block => "
            print('{:<157}'.format(c))       
            print('{:>157}'.format(output))  
            print('The code block printout => : ')
            block = ast.parse(c, mode='exec')
            exec(compile(block, '<string>', mode='exec'), globals().update(env))
        
        # handle assignment: 2. when = occur before if; 1. when no if only =
        elif ("=" in c and "if" not in c) or ("=" in c and c.find("=") < c.find("if")): # make sure assignment and !== and == are differentiated
            
            # print('k' in locals())
            exec(c, globals().update(env)) 
            # print('k' in locals())
            variable = c.partition(" = ")[0]
            # print(f"{c} => {variable}: {eval(variable)}")
            output = f"{c} => {variable}: {eval(variable)}"
            print('{:>157}'.format(output))       
            
        # handle if statement
        # Note: do insert code like this : `if abc == def: print(abc)`, print is a must
        elif "if" in c: 
            cond = re.search('if (.*?):', c).group(1)
            
            # when code in string is like 'if abc == def:'
            if c.endswith(':'):
                
                # print ... 
                # print(f"{c} => {cond}: {eval(cond)}")      
                output = f"{c} => {cond}: {eval(cond)}"
                print('{:>157}'.format(output))
                
            # when code in string is like 'if abc == def: print(...)'
            else: 
                # if the cond is true, then print ...
                if eval(cond):
                    
                    # "if abc == def: print(abc)".split(': ', 2)[1] to get 'print(abc)'
                    printc = c.split(': ', 1)[1]
                    # print(f"{c} => {printc} : ")
                    output = f"{c} => {printc} : "
                    print('{:>157}'.format(output))      
                    exec(c, globals().update(env))
                    
                # if cond is false, then print ...
                else: 
                    # print(f"{c} => {cond}: {eval(cond)}")
                    output = f"{c} => {cond}: {eval(cond)}"
                    print('{:>157}'.format(output))   
                
                
        # handle for in statement
        elif "for " in c and " in " in c:           
            
            # in example like 'for k, v in abc:' or `for k, v in abc: print(...)`, if abc is empty
            # step 1: access abc
            # get the substring between 'in ' and ':', which is like 'abc'
            abc = re.search('in (.*?):', c).group(1)
            # if abc is empty dict or list: print and pass
            if not bool(eval(abc)): 
                # print(f'{c} => {abc} is an emtpy {type(eval(abc))}')
                output = f'{c} => {abc} is an emtpy {type(eval(abc))}'
                print('{:>157}'.format(output))   
                continue 
                # The break statement can be used if you need to break out of a for or while loop and move onto the next section of code.
                # The continue statement can be used if you need to skip the current iteration of a for or while loop and move onto the next iteration.
            
            # if the code in string is like 'for k, v in abc:', there is no more code after `:`
            if c.endswith(':'):
                
                # get the substring between 'for ' and ' in', which is like 'k, v'
                variables = re.search('for (.*?) in', c).group(1)
                
                # if variables has a substring like ', ' inside
                if (',') in variables: 
                    
                    # split it by ', ' into a list of substrings
                    vl = variables.split(',')
                    key = vl[0]
                    value = vl[1]
                    
                    # make sure key and value will get evaluated first before exec run
                    # printc is for exec to run
                    printc = "print(f'{key}:{eval(key)}, {type(eval(key))} ; {value}:{eval(value)}, {type(eval(value))}')" 
                    # printmsg is for reader to understand with ease
                    printmsg = "print(f'key: {key}, {type(key)} ; value: {value}, {type(value)}')"
                    c1 = c + " " + printc
                    # print(f"{c} => {printmsg} : ")      
                    output = f"{c} => {printmsg} : "
                    print('{:>157}'.format(output))   
                    exec(c1, globals().update(env))
                
                else:
                    printc = "print(f'{variables} : {eval(variables)}')"
                    printmsg = "print(f'i : {variables}')"
                    c1 = c + " " + printc
                    # print(f"{c} => {printmsg} : ")     
                    output = f"{c} => {printmsg} : "
                    print('{:>157}'.format(output))   
                    exec(c1, globals().update(env))
                    
            # if the code in string is like 'for k, v in abc: print(abc)'
            else:                 
                # "for k, v in abc: print(k)".split(': ', 1)[1] to get 'print(k)'
                printc = c.split(': ', 1)[1]
                # print(f"{c} => {printc} : ")
                output = f"{c} => {printc} : "
                print('{:>157}'.format(output))   
                exec(c, globals().update(env)) # we can't use eval to run `for in` loop, but exec can.
            ### Note: we shall not use the expression like `for k, v in abc print(abc)`
            ### Note: we shall not use the expression like `for k, v in abc if k == def`
        
        
        # handle evaluation
        else: 
            # print(f"{c} => {c} : {eval(c, globals().update(env))}") 
            output = f"{c} => {c} : {eval(c, globals().update(env))}"
            print('{:>157}'.format(output))   
            
        # the benefit of using global().update(env) is 
        # to ensure we don't need to include the same env fo

# %% ../00_core.ipynb 254
class Fastdb():
    "Create a Fastdebug class which has two functionalities: dbprint and print."
    def __init__(self, 
                 src, # name of src code you are exploring
                 db=False): # db=True will run some debugging prints
        self.orisrc = src
        self.margin = 157
        self.outenv = src.__globals__
        self.cmts = {}
        if db:
            print(f"self.orisrc: {self.orisrc.__name__} is self.outenv['{self.orisrc.__name__}']: {self.orisrc is self.outenv[self.orisrc.__name__]}")

# %% ../00_core.ipynb 274
@patch
def dbprint(self:Fastdb, 
            idxsrc:int, # idx of a srcline under investigation, can only be int
            cmt:str, # comment added to the srcline
            *codes, # a list of expressions (str) you write to be evaluated above the srcline
            expand:int=2, # span 2 lines of srcode up and down from the srcline investigated
            showdbsrc:bool=False): # display dbsrc
    "Add comment and evaluate custom (single or multi lines) expressions to any srcline of the source code \
you are investigating. Run exec on the entire srcode with added expressions (dbsrc), so that dbsrc is callable."
    self.goback() # refresh 
    src = self.orisrc
    if type(idxsrc) == int: self.cmts.update({idxsrc: cmt})

    printsrc(src, idxsrc, cmt, expand)
    if showdbsrc:
        print('{:-<60}'.format(colorize("print selected srcline with expands above", color="y")))
    
    dbsrc = ""
    indent = 4

    lst = inspect.getsource(src).split('\n')
    if not bool(lst[-1]): lst = lst[:-1]

    newlst = []
    for i in codes: # no matter whether there is "" or "  " in the front or in the middle of codes
        if bool(i.strip()): newlst.append(i)
    codes = newlst

    srclines = ""
    if type(idxsrc) == int:
        srclines = lst[idxsrc]
    else:
        raise TypeError("decode must be an integer.")

    for idx, l in zip(range(len(lst)), lst):

        if bool(l.strip()) and l.strip() in srclines and idx == idxsrc:

            if len(codes) > 0: # no codes, no dbprintinsert
                numindent = len(l) - len(l.lstrip()) # make sure indent not messed up by trailing spaces
                dbcodes = "dbprintinsert("
                count = 1
                for c in codes:
                    if count == len(codes):
                        dbcodes = dbcodes + '"' + c + '"' + "," + "env=g" + ")"
                    else:
                        dbcodes = dbcodes + '"' + c + '"' + ","
                    count = count + 1

                dbsrc = dbsrc + " "*numindent + "g = locals()" + '\n'
                dbsrc = dbsrc + " "*numindent + dbcodes + '\n'
                dbsrc = dbsrc + l + '\n'     
            else:
                dbsrc = dbsrc + l + '\n'                

        elif bool(l.strip()) and idx + 1 == len(lst):
            dbsrc = dbsrc + l
        else: # make sure this printout is identical to the printsrc output
            dbsrc = dbsrc + l + '\n'

    if showdbsrc: # added to debug
        print('{:-<60}'.format(colorize("showdbsrc=Start", color="y")))
        totallen = 157
        lenidx = 5
        dblst = dbsrc.split('\n')
        idxsrcline = None
        for idx, l in zip(range(len(dblst)), dblst):
            lenl = len(l)
#             if "dbprintinsert" in l: 
            if l.strip().startswith("dbprintinsert"): 
                print(l + "="*(totallen-lenl-lenidx) + "(db)")
                idxsrcline = idx            
            elif not bool(l.strip()):
                if bool(idxsrcline) and idx > idxsrcline:
                    idx = idx - 1                
                print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
            else:
                if bool(idxsrcline) and idx > idxsrcline:
                    idx = idx - 1                
                print(l + "-"*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                
        print(f"locals() keys: {list(locals().keys())}")
        print(f"before exec, self.orisrc.__name__: {self.orisrc.__name__} is self.outenv['{self.orisrc.__name__}']: {self.orisrc is self.outenv[self.orisrc.__name__]}")
        print(f"before exec, self.orisrc.__name__: {self.orisrc.__name__} is : {self.orisrc}")
        names = self.orisrc.__qualname__.split('.')        
        if len(names) == 2:
            print('{:-<60}'.format(colorize("the src is a method of a class", color="y")))
            clsname = names[0]
            methodname = names[1]
            print(f"before exec, is methodname: {methodname} in locals(): {methodname in locals()}")
            print(f"before exec, is clsname: {clsname} in locals(): {clsname in locals()}")
            print(f"before exec, is self.orisrc.__qualname__: {self.orisrc.__qualname__} in locals(): {self.orisrc.__qualname__ in locals()}")
            print(f"before exec, is methodname: {methodname} in self.outenv: {methodname in self.outenv}")
            print(f"before exec, is clsname: {clsname} in self.outenv: {clsname in self.outenv}")
            print(f"before exec, is self.orisrc.__qualname__: {self.orisrc.__qualname__} in self.outenv: {self.orisrc.__qualname__ in self.outenv}")
            expr = "self.outenv[" + "'" + clsname + "']." + methodname
            print(f"expr: {expr}")
            expr1 = "self.outenv[" + "'" + methodname + "']"
            print(f"expr1: {expr1}")
            print(f"inspect.getsourcefile('{expr}') == '<string>': {True if inspect.getsourcefile(eval(expr)) == '<string>' else inspect.getsourcefile(eval(expr))}")
            print(f"self.outenv['{methodname}']: {eval(expr1)}")
            print(f"self.orisrc.__name__: {self.orisrc.__name__} is {expr}: {self.orisrc is eval(expr)}")
            
            
    exec(dbsrc, globals().update(self.outenv)) # when dbsrc is a method, it will update as part of a class
    if showdbsrc:
        print('{:-<60}'.format(colorize("exec on dbsrc above", color="y")))
    
    
    if showdbsrc: 
        print(f"locals() keys: {list(locals().keys())}")
        if len(names) == 2:
            print('{:-<60}'.format(colorize("the src is a method of a class", color="y")))            
            print(f"after exec, is methodname: {methodname} in locals(): {methodname in locals()}")
            print(f"after exec, is clsname: {clsname} in locals(): {clsname in locals()}")
            print(f"after exec, is self.orisrc.__qualname__: {self.orisrc.__qualname__} in locals(): {self.orisrc.__qualname__ in locals()}")
            print(f"after exec, is methodname: {methodname} in self.outenv: {methodname in self.outenv}")
            print(f"after exec, is clsname: {clsname} in self.outenv: {clsname in self.outenv}")
            print(f"after exec, is self.orisrc.__qualname__: {self.orisrc.__qualname__} in self.outenv: {self.orisrc.__qualname__ in self.outenv}")
            print(f"inspect.getsourcefile({expr}) == '<string>': {True if inspect.getsourcefile(eval(expr)) == '<string>' else inspect.getsourcefile(eval(expr))}")
            print(f"self.outenv['{methodname}']: {eval(expr1)}")
            print(f"self.orisrc.__name__: {self.orisrc.__name__} is {expr}: {self.orisrc is eval(expr)}")            
        print(f'self.orisrc.__name__: {self.orisrc.__name__}')
        print(f"locals()['{self.orisrc.__name__}']: {locals()[self.orisrc.__name__]}")

    
    if showdbsrc:
        print(f"after exec and before self.outenv update, self.outenv['{self.orisrc.__name__}'] is locals()['{self.orisrc.__name__}']: {locals()[self.orisrc.__name__] is self.outenv[self.orisrc.__name__]}")
        print(f"after exec and before self.outenv update, self.orisrc.__name__: {self.orisrc.__name__} is self.outenv['{self.orisrc.__name__}']: {self.orisrc is self.outenv[self.orisrc.__name__]}")
        print(f"after exec and before self.outenv update, self.orisrc.__name__: {self.orisrc.__name__} is locals()['{self.orisrc.__name__}']: {self.orisrc is locals()[self.orisrc.__name__]}")
    # Important! update dbsrc to module.func, fu.whatinside, inspect._signature_from_callable
    self.outenv.update(locals()) 
#     self.outenv[self.orisrc.__name__] = locals()[self.orisrc.__name__]
#     return locals()[self.orisrc.__name__]
    if showdbsrc:
        print(f"after update self.outenv, self.orisrc.__name__: {self.orisrc.__name__} is self.outenv['{self.orisrc.__name__}']: {self.orisrc is self.outenv[self.orisrc.__name__]}")
        exec("import " + self.orisrc.__module__.split('.')[0])
        print(f"after update self.outenv, self.orisrc.__name__: {self.orisrc.__name__} is {self.orisrc.__module__}.{self.orisrc.__name__}: {self.orisrc is eval(self.orisrc.__module__ + '.' + self.orisrc.__name__, {}, locals())}")
        print(f"after update self.outenv, self.outenv['{self.orisrc.__name__}'] is {self.orisrc.__module__}.{self.orisrc.__name__}: {self.outenv[self.orisrc.__name__] is eval(self.orisrc.__module__ + '.' + self.orisrc.__name__, {}, locals())}")        
        print(f"Therefore, to use dbsrc we must use self.outenv['{self.orisrc.__name__}'], or {self.orisrc.__module__}.{self.orisrc.__name__}")
        print('{:-<60}'.format(colorize("showdbsrc=End", color="y")))

        

# %% ../00_core.ipynb 275
@patch
def print(self:Fastdb, 
            maxlines:int=33, # maximum num of lines per page
            part:int=0): # if the src is more than 33 lines, then divide the src by 33 into a few parts
    "Print the source code in whole or parts with idx and comments you added with dbprint along the way."

    totallen = 157
    lenidx = 5
    lspace = 10
    lstsrc = inspect.getsource(self.orisrc).split('\n')
    numparts = len(lstsrc) // 33 + 1 if len(lstsrc) % 33 != 0 else len(lstsrc) // 33
    cmts = self.cmts
    if part == 0: 
        for idx, l in zip(range(len(lstsrc)), lstsrc):
            lenl = len(l)

            if not bool(l.strip()):
                print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")

            elif lenl + lspace >= 100:
                if bool(cmts):
                    cmtidx = [cmt[0] for cmt in list(cmts.items())]
                    if idx in cmtidx:
                        print(l + " # " + cmts[idx] + " "*(totallen-lenl-lenidx-len(cmts[idx])-3) + "(" + str(idx) + ")")
                    else:
                        print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                else: 
                    print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")

            else:


                if bool(cmts):
                    cmtidx = [cmt[0] for cmt in list(cmts.items())]
                    if idx in cmtidx:
                        print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})" + " # " + cmts[idx]))
                    else:
                        print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})"))                                                      

                else:
                    print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})"))                 

    for p in range(numparts):
        for idx, l in zip(range(len(lstsrc)), lstsrc):

            if (maxlines*p <= idx < maxlines*(p+1) and p+1 == part):
                lenl = len(l)
                if not bool(l.strip()):
                    print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                elif lenl + lspace >= 100:
                    if bool(cmts):
                        cmtidx = [cmt[0] for cmt in list(cmts.items())]
                        if idx in cmtidx:
                            print(l + " # " + cmts[idx] + " "*(totallen-lenl-lenidx-len(cmts[idx])-3) + "(" + str(idx) + ")")
                        else:
                            print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                    else: 
                        print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")


                else:

                    if bool(cmts):
                        cmtidx = [cmt[0] for cmt in list(cmts.items())]
                        if idx in cmtidx:
                            print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})" + " # " + cmts[idx]))
                        else:
                            print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})"))                                                          

                    else:
                        print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})"))                      

            if (idx == maxlines*(p+1) or idx == len(lstsrc) - 1) and p+1 == part:
                print('{:>157}'.format(f"part No.{p+1} out of {numparts} parts"))
                return

# %% ../00_core.ipynb 277
@patch
def goback(self:Fastdb):
    "Return src back to original state."
    self.outenv[self.orisrc.__name__] = self.orisrc

# %% ../00_core.ipynb 284
@patch
def explore(self:Fastdb, 
            idxsrc:int, # idxsrc can be an int or a list of int
            cmt:str, # comment can be a string or a list of strings
            showdbsrc:bool=False): # display dbsrc
    "insert 'import ipdb; ipdb.set_trace()' above srcline of idx to create dbsrc, and exec on dbsrc"
    self.goback()
    src = self.orisrc

#     printsrc(src, idxsrc, cmt)
    if showdbsrc:
        print('{:-<60}'.format(colorize("print selected srcline with expands above", color="y")))
    
    dbsrc = ""
    indent = 4

    lst = inspect.getsource(src).split('\n')
    if not bool(lst[-1]): lst = lst[:-1]

    srclines = None
    idxlst = None
    if type(idxsrc) == int:
        srclines = lst[idxsrc]
    elif type(idxsrc) == list:
        idxlst = idxsrc
    else:
        raise TypeError("decode must be an integer or a list.")

    for idx, l in zip(range(len(lst)), lst):

        if bool(l.strip()) and type(idxsrc) == int and idx == idxsrc:
            numindent = len(l) - len(l.lstrip()) 
            dbcodes = "import ipdb; ipdb.set_trace()"
            dbsrc = dbsrc + " "*numindent + dbcodes + '\n'
            dbsrc = dbsrc + l + '\n'     
        elif type(idxsrc) == list and idx in idxlst:
            numindent = len(l) - len(l.lstrip()) 
            dbcodes = "import ipdb; ipdb.set_trace()"
            dbsrc = dbsrc + " "*numindent + dbcodes + '\n'
            dbsrc = dbsrc + l + '\n'  
            idxlst.remove(idx)
        elif bool(l.strip()) and idx + 1 == len(lst):
            dbsrc = dbsrc + l
        else: # make sure this printout is identical to the printsrc output
            dbsrc = dbsrc + l + '\n'

    if showdbsrc: # added to debug
        print('{:-<60}'.format(colorize("print out dbsrc", color="y")))
        totallen = 157
        lenidx = 5
        dblst = dbsrc.split('\n')
        idxsrcline = None
        idxcount = 0
        for idx, l in zip(range(len(dblst)), dblst):
            lenl = len(l)
#             if "dbprintinsert" in l: 
            if l.strip().startswith("import ipdb"): 
                print(l + "="*(totallen-lenl-lenidx) + "(db)")
                idxsrcline = idx
                idxcount = idxcount + 1
            elif not bool(l.strip()):
                if bool(idxsrcline) and idx > idxsrcline:
                    idx = idx - idxcount
                print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
            else:
                if bool(idxsrcline) and idx > idxsrcline:
                    idx = idx - idxcount
                print(l + "-"*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
        
        print(f"locals() keys: {list(locals().keys())}")
        print(f"before exec, self.orisrc.__name__: {self.orisrc.__name__} is self.outenv['{self.orisrc.__name__}']: {self.orisrc is self.outenv[self.orisrc.__name__]}")
        print(f"before exec, self.orisrc.__name__: {self.orisrc.__name__} is : {self.orisrc}")
        names = self.orisrc.__qualname__.split('.')        
        if len(names) == 2:
            print('{:-<60}'.format(colorize("the src is a method of a class", color="y")))
            clsname = names[0]
            methodname = names[1]
            print(f"before exec, is methodname: {methodname} in locals(): {methodname in locals()}")
            print(f"before exec, is clsname: {clsname} in locals(): {clsname in locals()}")
            print(f"before exec, is self.orisrc.__qualname__: {self.orisrc.__qualname__} in locals(): {self.orisrc.__qualname__ in locals()}")
            print(f"before exec, is methodname: {methodname} in self.outenv: {methodname in self.outenv}")
            print(f"before exec, is clsname: {clsname} in self.outenv: {clsname in self.outenv}")
            print(f"before exec, is self.orisrc.__qualname__: {self.orisrc.__qualname__} in self.outenv: {self.orisrc.__qualname__ in self.outenv}")
            expr = "self.outenv[" + "'" + clsname + "']." + methodname
            print(f"expr: {expr}")
            expr1 = "self.outenv[" + "'" + methodname + "']"
            print(f"expr1: {expr1}")
            print(f"inspect.getsourcefile('{expr}') == '<string>': {True if inspect.getsourcefile(eval(expr)) == '<string>' else inspect.getsourcefile(eval(expr))}")
            print(f"self.outenv['{methodname}']: {eval(expr1)}")
            print(f"self.orisrc.__name__: {self.orisrc.__name__} is {expr}: {self.orisrc is eval(expr)}")

    file_name ='/tmp/' + self.orisrc.__name__ + '.py' # learn about /tmp folder https://www.fosslinux.com/41739/linux-tmp-directory-everything-you-need-to-know.htm
    with open(file_name, 'w') as f:
        f.write(dbsrc)
    code = compile(dbsrc, file_name, 'exec')
    exec(code, globals().update(self.outenv)) # when dbsrc is a method, it will update as part of a class
    if showdbsrc:
        print('{:-<60}'.format(colorize("exec on dbsrc above", color="y")))
    
    
    if showdbsrc: 
        print(f"locals() keys: {list(locals().keys())}")
        if len(names) == 2:
            print('{:-<60}'.format(colorize("the src is a method of a class", color="y")))            
            print(f"after exec, is methodname: {methodname} in locals(): {methodname in locals()}")
            print(f"after exec, is clsname: {clsname} in locals(): {clsname in locals()}")
            print(f"after exec, is self.orisrc.__qualname__: {self.orisrc.__qualname__} in locals(): {self.orisrc.__qualname__ in locals()}")
            print(f"after exec, is methodname: {methodname} in self.outenv: {methodname in self.outenv}")
            print(f"after exec, is clsname: {clsname} in self.outenv: {clsname in self.outenv}")
            print(f"after exec, is self.orisrc.__qualname__: {self.orisrc.__qualname__} in self.outenv: {self.orisrc.__qualname__ in self.outenv}")
            print(f"inspect.getsourcefile({expr}) == '<string>': {True if inspect.getsourcefile(eval(expr)) == '<string>' else inspect.getsourcefile(eval(expr))}")
            print(f"self.outenv['{methodname}']: {eval(expr1)}")
            print(f"self.orisrc.__name__: {self.orisrc.__name__} is {expr}: {self.orisrc is eval(expr)}")            
        print(f'self.orisrc.__name__: {self.orisrc.__name__}')
        print(f"locals()['{self.orisrc.__name__}']: {locals()[self.orisrc.__name__]}")

    
    if showdbsrc:
        print(f"after exec and before self.outenv update, self.outenv['{self.orisrc.__name__}'] is locals()['{self.orisrc.__name__}']: {locals()[self.orisrc.__name__] is self.outenv[self.orisrc.__name__]}")
        print(f"after exec and before self.outenv update, self.orisrc.__name__: {self.orisrc.__name__} is self.outenv['{self.orisrc.__name__}']: {self.orisrc is self.outenv[self.orisrc.__name__]}")
        print(f"after exec and before self.outenv update, self.orisrc.__name__: {self.orisrc.__name__} is locals()['{self.orisrc.__name__}']: {self.orisrc is locals()[self.orisrc.__name__]}")
    # Important! update dbsrc to module.func, fu.whatinside, inspect._signature_from_callable
    self.outenv.update(locals()) 
#     self.outenv[self.orisrc.__name__] = locals()[self.orisrc.__name__]
#     return locals()[self.orisrc.__name__]
    if showdbsrc:
        print(f"after update self.outenv, self.orisrc.__name__: {self.orisrc.__name__} is self.outenv['{self.orisrc.__name__}']: {self.orisrc is self.outenv[self.orisrc.__name__]}")
        exec("import " + self.orisrc.__module__.split('.')[0])
        print(f"after update self.outenv, self.orisrc.__name__: {self.orisrc.__name__} is {self.orisrc.__module__}.{self.orisrc.__name__}: {self.orisrc is eval(self.orisrc.__module__ + '.' + self.orisrc.__name__, {}, locals())}")
        print(f"after update self.outenv, self.outenv['{self.orisrc.__name__}'] is {self.orisrc.__module__}.{self.orisrc.__name__}: {self.outenv[self.orisrc.__name__] is eval(self.orisrc.__module__ + '.' + self.orisrc.__name__, {}, locals())}")        
        print(f"Therefore, to use dbsrc we must use self.outenv['{self.orisrc.__name__}'], or {self.orisrc.__module__}.{self.orisrc.__name__}")
        print('{:-<60}'.format(colorize("showdbsrc=End", color="y")))

        

# %% ../00_core.ipynb 295
def reliveonce(func, # the current func
               oldfunc:str, # the old version of func in string
               alive:bool=True, # True to bring old to live, False to return back to normal
               db=False): # True, to print for debugging
    "Replace current version of srcode with older version, and back to normal"
    if alive:
        safety = func
        block = ast.parse(oldfunc, mode='exec')
        exec(compile(block, '<string>', mode='exec'), globals().update(func.__globals__))
        if db:
            print(f"after exec: list(locals().keys()): {list(locals().keys())}")
            print(f"before update: inspect.getsourcefile(func.__globals__[func.__name__]): {inspect.getsourcefile(func.__globals__[func.__name__])}")
        
        # update the old version of func from locals() into func.__globals__, so that outside this reliveonce function, the old func can be used
        func.__globals__.update(locals())
        if db:
            print(f"after update: inspect.getsourcefile(func.__globals__[func.__name__]): {inspect.getsourcefile(func.__globals__[func.__name__])}")
    else:
        func.__globals__[func.__name__] = func.__globals__['safety']
