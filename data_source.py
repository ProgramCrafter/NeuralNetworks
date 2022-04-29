import math
import png

class IDataSource:
  def __init__(self, path):     pass
  def extract_data(self, case): pass
  def wanted(self, case):       pass
  def cases(self):              pass

class XORDataSource(IDataSource):
  def __init__(self): pass
  
  def cases(self):
    return 4
  
  def extract_data(self, case):
    return [case // 2, case % 2]
  
  def wanted(self, case):
    return [case in (1, 2)]
