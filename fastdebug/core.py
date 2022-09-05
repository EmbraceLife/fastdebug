# AUTOGENERATED! DO NOT EDIT! File to edit: ../00_core.ipynb.

# %% auto 0
__all__ = ['defaults', 'dbcolors', 'colorize', 'strip_ansi', 'alignright', 'printsrcwithidx', 'printsrclinewithidx', 'printsrc',
           'dbprint', 'dbprintinsert', 'printrunsrclines', 'Fastdb']

# %% ../00_core.ipynb 4
defaults = type('defaults', (object,), {'margin': 157, # align to the right by 157
                                        'orisrc': None, # keep a copy of original official src code
                                        'outenv': globals(), # outside global env
                                        'cmts': {} # a dict to store idx and cmt
                                        # 'eg': None, # examples
                                        # 'src': None, # official src
                                       }) 

# %% ../00_core.ipynb 19
from pprint import pprint

# %% ../00_core.ipynb 30
import inspect

# %% ../00_core.ipynb 49
class dbcolors:
    g = '\033[92m' #GREEN
    y = '\033[93m' #YELLOW
    r = '\033[91m' #RED
    reset = '\033[0m' #RESET COLOR

# %% ../00_core.ipynb 50
def colorize(cmt, color:str=None):
    if color == "g":
        return dbcolors.g + cmt + dbcolors.reset
    elif color == "y":
        return dbcolors.y + cmt + dbcolors.reset
    elif color == "r":
        return dbcolors.r + cmt + dbcolors.reset
    else: 
        return cmt

# %% ../00_core.ipynb 54
import re

# %% ../00_core.ipynb 55
def strip_ansi(source):
    return re.sub(r'\033\[(\d|;)+?m', '', source)

# %% ../00_core.ipynb 56
def alignright(blocks):
    lst = blocks.split('\n')
    maxlen = max(map(lambda l : len(strip_ansi(l)) , lst ))
    indent = defaults.margin - maxlen
    for l in lst:
        print(' '*indent + format(l))

# %% ../00_core.ipynb 65
def printsrcwithidx(src, 
                    maxlines:int=33, # maximum num of lines per page
                    part:int=0): # if the src is more than 33 lines, then divide the src by 33 into a few parts
    totallen = 157
    lenidx = 5
    lspace = 10
    lstsrc = inspect.getsource(src).split('\n')
    numparts = len(lstsrc) // 33 + 1 if len(lstsrc) % 33 != 0 else len(lstsrc) // 33
    # cmts = {5:"this is me", 111:"this is me", 14:"this is you this is you this is you this is you this is you this is you this is you this is you "}
    cmts = defaults.cmts
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

# %% ../00_core.ipynb 71
import inspect

# %% ../00_core.ipynb 86
def printsrclinewithidx(idx, l, fill=" "):
    totallen = 157
    lenidx = 5
    lenl = len(l)
    print(l + fill*(totallen-lenl-lenidx) + "(" + str(idx) + ")")

# %% ../00_core.ipynb 93
def printsrc(src, # name of src code such as foo, or delegates
             dbcode, # string of codes or int of code idx number
             cmt,
             expand:int=2): # expand the codes around the srcline under investigation
    
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


# %% ../00_core.ipynb 126
def dbprint(src, # the src func name, e.g., foo
            dbcode, # the srclines under investigation, can be either string or int
            cmt:str, # comment
            *codes, # a list of dbcodes
            expand:int=2, # span 2 lines of srcode up and down from the srcline investigated
            env={}, # out environment
            showdbsrc=False): # print out dbsrc or not
    "Insert dbcodes under srclines under investigation, and create a new dbsrc function to replace the official one"

    # make sure the original official src is kept safe and used whenever dbprint is used
    if defaults.orisrc == None:
        defaults.orisrc = src
    else: 
        src = defaults.orisrc
        
    if type(dbcode) == int: defaults.cmts.update({dbcode: cmt})

    printsrc(src, dbcode, cmt, expand)

    dbsrc = ""
    indent = 4
    onedbprint = False

    lst = inspect.getsource(src).split('\n')
    if not bool(lst[-1]): lst = lst[:-1]

    srclines = ""
    if type(dbcode) == int:
        srclines = lst[dbcode]
    else:
        srclines = dbcode

    for idx, l in zip(range(len(lst)), lst):
        # make sure the line with correct idx is debugged
        if bool(l.strip()) and l.strip() in srclines and idx == dbcode: 

            numindent = len(l) - len(l.strip())
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

        elif bool(l.strip()) and idx + 1 == len(lst):
            dbsrc = dbsrc + l

        elif bool(l.strip()): # make sure pure indentation + \n is ignored
            dbsrc = dbsrc + l + '\n'
  

    if showdbsrc: # added to debug
        for l in dbsrc.split('\n'):
            print(l)
    
    exec(dbsrc, globals().update(env)) # make sure b can access lst from above

    env.update(locals())

    return locals()[defaults.orisrc.__name__]


# %% ../00_core.ipynb 145
import ast

# %% ../00_core.ipynb 146
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

# %% ../00_core.ipynb 200
def printrunsrclines(func, example):
    srclines = inspect.getsource(func).split('\n')
    dbsrc = ""
    indent = 4

    for idx, l in zip(range(len(srclines)), srclines):
        numindent = len(l) - len(l.strip())
        addline = f"srcidx.append({idx})"

        if "if" in l or "else" in l or "for" in l or "def" in l:
            numindent = numindent + indent

        if bool(l) and not l.strip().startswith('#') and not (l.strip().startswith('"') and l.strip().endswith('"')): # ignore/remove pure quotations or docs
            dbsrc = dbsrc + l + "\n" + " "*numindent + addline + "\n"  # add srcidx.append(idx) to each line

    pprint(dbsrc)
    srcidx = [] 
    exec(dbsrc, globals().update(locals()))    
    exec(example) # now we can use foo as the new foo 
    print(srcidx)

    for idx, l in zip(range(len(srclines)), srclines):
        if idx in srcidx or "for" in l or "if" in l or "else" in l:
            print(l)

# %% ../00_core.ipynb 214
def printrunsrclines(src, example, env):
    srclst = inspect.getsource(src).split('\n')
    dbsrc = ""
    indent = 4
    bracketidx = []
    bracketindent = 0
    ifelseidx = []
    ifelseindent = 0
    
    for idx, l in zip(range(len(srclst)), srclst):
        numindent = len(l) - len(l.strip()) # how to strip only the left not the right?????
        addline = f"srcidx.append({idx})"

        if ("if " in l and l.strip().endswith(':')) or l.strip().endswith("else:") or ("elif " in l and l.strip().endswith(':')) \
        or  (l.strip().endswith(':') and "for " in l) or ("def " in l and l.strip().endswith(':')):
            numindent = numindent + indent

            
        
        if l.strip().startswith('"""'):
            dbsrc = dbsrc + l + '\n'
        elif srclst[idx - 1].strip().startswith('"""') and '"""' not in l.strip():
            dbsrc = dbsrc + l + '\n'
        elif idx <= len(srclst) - 2 and srclst[idx + 1].strip().startswith('"""') and '"""' not in l.strip():
            dbsrc = dbsrc + l + '\n'
            
        elif "{" in l and "}" not in l:
            bracketidx.append(idx)
            bracketindent = len(l) - len(l.strip())
            dbsrc = dbsrc + l + '\n'
        elif "}" in l and "{" not in l: 
            bracketidx.append(idx)
            addup = ""
            for i in bracketidx:
                line = f"srcidx.append({i})"
                addup = addup + " "*bracketindent + line + "\n"
            dbsrc = dbsrc + l + "\n" + addup
            
        elif (l.strip().startswith("if") or l.strip().startswith("elif")) and ":" in l and not l.strip().endswith(":") and ": #" not in l \
        and ("elif" in srclst[idx + 1] or "else" in srclst[idx + 1]):
            ifelseidx.append(idx)
            ifelseindent = len(l) - len(l.strip())
            dbsrc = dbsrc + l + '\n'
        elif l.strip().startswith("else") and ":" in l and not l.strip().endswith(":") and ": #" not in l:
            ifelseidx.append(idx)
            addup = ""
            for i in ifelseidx:
                line = f"srcidx.append({i})"
                addup = addup + " "*ifelseindent + line + "\n"
            dbsrc = dbsrc + l + "\n" + addup            
            
            
        elif bool(l.strip()) and not l.strip().startswith('#') \
        and not (l.strip().startswith('"') and l.strip().endswith('"')) \
        and not (l.strip().endswith(',') or ', #' in l or '): #' in l): 
            dbsrc = dbsrc + l + "\n" + " "*numindent + addline + "\n"  # add srcidx.append(idx) to each line
        else: 
            dbsrc = dbsrc + l + '\n'            

    pprint(dbsrc, width=157)
    srcidx = [] 
    exec(dbsrc, globals().update(env), locals())    
    exec(example) # now we can use foo as the new foo 
    print(srcidx)

    # pprint(srclines)
    for idx, l in zip(range(len(srclst)), srclst):
        if idx in srcidx or "for" in l or "if" in l or "else" in l:
            print(l)

# %% ../00_core.ipynb 235
class Fastdb():
    
    def __init__(self, 
                 src, # name of src code
                 env): # env = g, as g = globals()
        self.orisrc = src
        self.margin = 157
        self.outenv = env
        self.cmts = {}

        
    def dbprint(self, 
                # src, # the src func name, e.g., foo
                dbcode, # the srclines under investigation, can be either string or int
                cmt:str, # comment
                *codes, # a list of dbcodes
                expand:int=2, # span 2 lines of srcode up and down from the srcline investigated
                showdbsrc=False): # display dbsrc or not
        "Insert dbcodes under srclines under investigation, and create a new dbsrc function to replace the official one"

        src = self.orisrc
        if type(dbcode) == int: self.cmts.update({dbcode: cmt})

        printsrc(src, dbcode, cmt, expand)

        dbsrc = ""
        indent = 4
        onedbprint = False

        lst = inspect.getsource(src).split('\n')
        if not bool(lst[-1]): lst = lst[:-1]
        
        newlst = []
        for i in codes: # no matter whether there is "" or "  " in the front or in the middle of codes
            if bool(i.strip()): newlst.append(i)
        codes = newlst

        srclines = ""
        if type(dbcode) == int:
            srclines = lst[dbcode]
        else:
            srclines = dbcode

        for idx, l in zip(range(len(lst)), lst):

            if bool(l.strip()) and l.strip() in srclines and idx == dbcode: 

                if len(codes) > 0: # if the new codes is not empty
                    numindent = len(l) - len(l.strip())
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

                else:
                    dbsrc = dbsrc + l + '\n'                

            elif bool(l.strip()) and idx + 1 == len(lst):
                dbsrc = dbsrc + l

            elif bool(l.strip()): # make sure pure indentation + \n is ignored
                dbsrc = dbsrc + l + '\n'
                
        if showdbsrc: # added to debug
            for l in dbsrc.split('\n'):
                print(l)
                
        exec(dbsrc, globals().update(self.outenv)) # make sure b can access lst from above

        self.outenv.update(locals())

        return locals()[self.orisrc.__name__]
    
    
    def print(self, 
                maxlines:int=33, # maximum num of lines per page
                part:int=0): # if the src is more than 33 lines, then divide the src by 33 into a few parts
        totallen = 157
        lenidx = 5
        lspace = 10
        lstsrc = inspect.getsource(self.orisrc).split('\n')
        numparts = len(lstsrc) // 33 + 1 if len(lstsrc) % 33 != 0 else len(lstsrc) // 33
        # cmts = {5:"this is me", 111:"this is me", 14:"this is you this is you this is you this is you this is you this is you this is you this is you "}
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

# %% ../00_core.ipynb 237
class Fastdb():
    
    def __init__(self, 
                 src, # name of src code
                 env): # env = g, as g = globals()
        self.orisrc = src
        self.margin = 157
        self.outenv = env
        self.cmts = {}
        
    def dbprint(self, 
                # src, # the src func name, e.g., foo
                dbcode, # the srclines under investigation, can be either string or int
                cmt:str, # comment
                *codes, # a list of dbcodes
                expand:int=2, # span 2 lines of srcode up and down from the srcline investigated
                showdbsrc=False): # display dbsrc or not
        "Insert dbcodes under srclines under investigation, and create a new dbsrc function to replace the official one"

        src = self.orisrc
        if type(dbcode) == int: self.cmts.update({dbcode: cmt})

        printsrc(src, dbcode, cmt, expand)

        dbsrc = ""
        indent = 4
        onedbprint = False

        lst = inspect.getsource(src).split('\n')
        if not bool(lst[-1]): lst = lst[:-1]
        
        newlst = []
        for i in codes: # no matter whether there is "" or "  " in the front or in the middle of codes
            if bool(i.strip()): newlst.append(i)
        codes = newlst

        srclines = ""
        if type(dbcode) == int:
            srclines = lst[dbcode]
        else:
            srclines = dbcode

        for idx, l in zip(range(len(lst)), lst):

            if bool(l.strip()) and l.strip() in srclines and idx == dbcode: 

                if len(codes) > 0: # if the new codes is not empty
                    numindent = len(l) - len(l.strip())
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

                else:
                    dbsrc = dbsrc + l + '\n'                

            elif bool(l.strip()) and idx + 1 == len(lst):
                dbsrc = dbsrc + l

            elif bool(l.strip()): # make sure pure indentation + \n is ignored
                dbsrc = dbsrc + l + '\n'
                
        if showdbsrc: # added to debug
            for l in dbsrc.split('\n'):
                print(l)
                
        exec(dbsrc, globals().update(self.outenv)) # make sure b can access lst from above

        self.outenv.update(locals())

        return locals()[self.orisrc.__name__]
    
    
    def print(self, 
                maxlines:int=33, # maximum num of lines per page
                part:int=0): # if the src is more than 33 lines, then divide the src by 33 into a few parts
        totallen = 157
        lenidx = 5
        lspace = 10
        lstsrc = inspect.getsource(self.orisrc).split('\n')
        numparts = len(lstsrc) // 33 + 1 if len(lstsrc) % 33 != 0 else len(lstsrc) // 33
        # cmts = {5:"this is me", 111:"this is me", 14:"this is you this is you this is you this is you this is you this is you this is you this is you "}
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
