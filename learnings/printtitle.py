
@patch====================================================================================(0)       
def printtitle(self:Fastdb):==============================================================(1)       
    "print title which includes src name, line number under investigation, example."======(2)       
    if 'self.dbsrc' not in self.eg:=======================================================(3)       
        self.orieg = self.eg  # make sure self.orieg has no self inside===================(4)       
    print('{:=^157}'.format(f"     Investigating {colorize(self.orisrc.__name__, color='r')}     "))  # [91;1mhow to use :=<, :=>, :=^ with format to align text to left, right, and middle[0m;  (5)
    print('{:=^157}'.format(f"     on line {colorize(str(self.idxsrc), color='r')}     "))                                                              (6)
    print('{:=^157}'.format(f"     with example {colorize(self.orieg, color='r')}     "))                                                               (7)
    print()===============================================================================(8)       
                                                                                                                                                        (9)