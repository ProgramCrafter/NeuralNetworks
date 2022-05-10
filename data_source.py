import random
import math
import png

def flat(arr):
  for v in arr:
    yield from v

class IDataSource:
  def __init__(self, path):     pass
  def extract_data(self, case): pass
  def wanted(self, case):       pass
  def cases(self):              pass

class IconDataExtractor(IDataSource):
  def __init__(self, prefix, paths):
    self.images = [
      list(flat(png.Reader(prefix + path).read()[2])) for path in paths
    ]
  
  def extract_data(self, case):
    return [v / 256 for v in self.images[case]]
  
  def wanted(self, case):
    values = [0, 1, 1, 0] * 4
    return [values[case]]
  
  def cases(self):
    return 16
