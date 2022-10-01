# 0011_Fastdb


```
from fastdebug.utils import *
from fastdebug.core import *
```


<style>.container { width:100% !important; }</style>



```
fdb = Fastdb(Fastdb.printtitle)
```


```
fdb.docsrc(5, "how to use :=<, :=>, :=^ with format to align text to left, right, and middle")
fdb.print()
```

    ========================================================     Investigating [91;1mprinttitle[0m     ========================================================
    ===============================================================     on line [91;1m5[0m     ================================================================
    =============================================================     with example [91;1m[0m     ==============================================================
    
    [93;1mprint selected srcline with expands below[0m--------
        if 'self.dbsrc' not in self.eg:                                                                                                                     (3)
            self.orieg = self.eg  # make sure self.orieg has no self inside                                                                                 (4)
        print('{:=^157}'.format(f"     Investigating {colorize(self.orisrc.__name__, color='r')}     ")) ===================================================(5)
                                                                                    [91;1mhow to use :=<, :=>, :=^ with format to align text to left, right, and middle[0m
        print('{:=^157}'.format(f"     on line {colorize(str(self.idxsrc), color='r')}     "))                                                              (6)
        print('{:=^157}'.format(f"     with example {colorize(self.orieg, color='r')}     "))                                                               (7)
    ========================================================     Investigating [91;1mprinttitle[0m     ========================================================
    ===============================================================     on line [91;1m5[0m     ================================================================
    =============================================================     with example [91;1m[0m     ==============================================================
    
    @patch====================================================================================(0)       
    def printtitle(self:Fastdb):==============================================================(1)       
                                                                                                                                                            (2)
        if 'self.dbsrc' not in self.eg:=======================================================(3)       
            self.orieg = self.eg  # make sure self.orieg has no self inside===================(4)       
        print('{:=^157}'.format(f"     Investigating {colorize(self.orisrc.__name__, color='r')}     "))  # [35;1mhow to use :=<, :=>, :=^ with format to align text to left, right, and middle[0m;  (5)
        print('{:=^157}'.format(f"     on line {colorize(str(self.idxsrc), color='r')}     "))                                                              (6)
        print('{:=^157}'.format(f"     with example {colorize(self.orieg, color='r')}     "))                                                               (7)
        print()===============================================================================(8)       
                                                                                                                                                            (9)



```
fdb = Fastdb(Fastdb.snoop)
```


```
fdb.print()
```

    ==========================================================     Investigating [91;1msnoop[0m     ===========================================================
    ==============================================================     on line [91;1mNone[0m     ==============================================================
    =============================================================     with example [91;1m[0m     ==============================================================
    
    def create_snoop_from_string(self:Fastdb, db=False):======================================(0)       
        # learn about /tmp folder https://www.fosslinux.com/41739/linux-tmp-directory-everything-you-need-to-know.htm                                       (1)
        file_name ='/tmp/' + self.orisrc.__name__ + '.py' ====================================(2)       
        with open(file_name, 'w') as f:=======================================================(3)       
            f.write(self.dbsrcstr)============================================================(4)       
        code = compile(self.dbsrcstr, file_name, 'exec')======================================(5)       
    #             exec(dbsrc, locals(), self.egEnv)                ===========================(6)       
    #     exec(code, globals().update(self.outenv), locals()) # when dbsrc is a method, it will update as part of a class                                   (7)
        exec(code, globals().update(self.outenv)) # when dbsrc is a method, it will update as part of a class                                               (8)
        # store dbsrc func inside Fastdb obj==================================================(9)       
        self.dbsrc = locals()[self.orisrc.__name__]===========================================(10)      
                                                                                                                                                            (11)



```
fdb = Fastdb(Fastdb.create_explore_str)
```


```
fdb.print()
```

    ====================================================     Investigating [91;1mcreate_explore_str[0m     ====================================================
    ==============================================================     on line [91;1mNone[0m     ==============================================================
    =============================================================     with example [91;1m[0m     ==============================================================
    
    @patch====================================================================================(0)       
    def create_explore_str(self:Fastdb):======================================================(1)       
        dbsrc = ""============================================================================(2)       
        indent = 4============================================================================(3)       
                                                                                                                                                            (4)
        lst = inspect.getsource(self.orisrc).split('\n')======================================(5)       
        if not bool(lst[-1]): lst = lst[:-1]==================================================(6)       
                                                                                                                                                            (7)
        srclines = None=======================================================================(8)       
        idxlst = None=========================================================================(9)       
        if type(self.idxsrc) == int:==========================================================(10)      
            srclines = lst[self.idxsrc]=======================================================(11)      
        elif type(self.idxsrc) == list:=======================================================(12)      
            idxlst = self.idxsrc==============================================================(13)      
        else:=================================================================================(14)      
            raise TypeError("decode must be an integer or a list.")===========================(15)      
                                                                                                                                                            (16)
        for idx, l in zip(range(len(lst)), lst):==============================================(17)      
                                                                                                                                                            (18)
            if bool(l.strip()) and type(self.idxsrc) == int and idx == self.idxsrc:===========(19)      
                numindent = len(l) - len(l.lstrip()) =========================================(20)      
                dbcodes = "import ipdb; ipdb.set_trace()"=====================================(21)      
                dbsrc = dbsrc + " "*numindent + dbcodes + '\n'================================(22)      
                dbsrc = dbsrc + l + '\n'     =================================================(23)      
            elif type(self.idxsrc) == list and idx in idxlst:=================================(24)      
                numindent = len(l) - len(l.lstrip()) =========================================(25)      
                dbcodes = "import ipdb; ipdb.set_trace()"=====================================(26)      
                dbsrc = dbsrc + " "*numindent + dbcodes + '\n'================================(27)      
                dbsrc = dbsrc + l + '\n'  ====================================================(28)      
                idxlst.remove(idx)============================================================(29)      
            elif bool(l.strip()) and idx + 1 == len(lst):=====================================(30)      
                dbsrc = dbsrc + l=============================================================(31)      
            else: # make sure this printout is identical to the printsrc output===============(32)      
                dbsrc = dbsrc + l + '\n'======================================================(33)      
                                                                                                                                                            (34)
        self.dbsrcstr = dbsrc=================================================================(35)      
                                                                                                                                                            (36)



```
fastview(Fastdb.printtitle)
```

    @patch====================================================================================(0)       
    def printtitle(self:Fastdb):==============================================================(1)       
                                                                                                                                                            (2)
        if 'self.dbsrc' not in self.eg:=======================================================(3)       
            self.orieg = self.eg  # make sure self.orieg has no self inside===================(4)       
        print('{:=^157}'.format(f"     Investigating {colorize(self.orisrc.__name__, color='r')}     "))  # [93;1mhow to use :=<, :=>, :=^ with format to align text to left, right, and middle[0m;  (5)
        print('{:=^157}'.format(f"     on line {colorize(str(self.idxsrc), color='r')}     "))                                                              (6)
        print('{:=^157}'.format(f"     with example {colorize(self.orieg, color='r')}     "))                                                               (7)
        print()===============================================================================(8)       
                                                                                                                                                            (9)



```

```
