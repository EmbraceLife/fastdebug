
@patch====================================================================================(0)       
def create_explore_str(self:Fastdb):======================================================(1)       
    "create the explore dbsrc string"=====================================================(2)       
    dbsrc = ""============================================================================(3)       
    indent = 4============================================================================(4)       
                                                                                                                                                        (5)
    lst = inspect.getsource(self.orisrc).split('\n')======================================(6)       
    if not bool(lst[-1]): lst = lst[:-1]==================================================(7)       
                                                                                                                                                        (8)
    srclines = None=======================================================================(9)       
    idxlst = None=========================================================================(10)      
    if type(self.idxsrc) == int:==========================================================(11)      
        srclines = lst[self.idxsrc]=======================================================(12)      
    elif type(self.idxsrc) == list:=======================================================(13)      
        idxlst = self.idxsrc==============================================================(14)      
    else:=================================================================================(15)      
        raise TypeError("decode must be an integer or a list.")===========================(16)      
                                                                                                                                                        (17)
    for idx, l in zip(range(len(lst)), lst):==============================================(18)      
                                                                                                                                                        (19)
        if bool(l.strip()) and type(self.idxsrc) == int and idx == self.idxsrc:===========(20)      
            numindent = len(l) - len(l.lstrip()) =========================================(21)      
            dbcodes = "import ipdb; ipdb.set_trace()"=====================================(22)      
            dbsrc = dbsrc + " "*numindent + dbcodes + '\n'================================(23)      
            dbsrc = dbsrc + l + '\n'     =================================================(24)      
        elif type(self.idxsrc) == list and idx in idxlst:=================================(25)      
            numindent = len(l) - len(l.lstrip()) =========================================(26)      
            dbcodes = "import ipdb; ipdb.set_trace()"=====================================(27)      
            dbsrc = dbsrc + " "*numindent + dbcodes + '\n'================================(28)      
            dbsrc = dbsrc + l + '\n'  ====================================================(29)      
            idxlst.remove(idx)============================================================(30)      
        elif bool(l.strip()) and idx + 1 == len(lst):=====================================(31)      
            dbsrc = dbsrc + l=============================================================(32)      
        else: # make sure this printout is identical to the printsrc output===============(33)      
            dbsrc = dbsrc + l + '\n'======================================================(34)      
                                                                                                                                                        (35)
    self.dbsrcstr = dbsrc=================================================================(36)      
                                                                                                                                                        (37)