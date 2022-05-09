import random
import math
import png

class IDataSource:
  def __init__(self, path):     pass
  def extract_data(self, case): pass
  def wanted(self, case):       pass
  def cases(self):              pass

class HsvDataExtractor(IDataSource):
  def __init__(self, path):
    mtx = png.Reader(path).read()[2]
    self.image = [list(row) for row in mtx]
    
    random.seed(0x14609A25)
    self.next_cases = list(range(self.cases()))
    random.shuffle(self.next_cases)
    
    k = 0.7
    self.angles = [case * (360 // self.cases()) * math.pi / 180
      for case in range(self.cases())]
    self.row_refs = [self.image[int(256 * (math.sin(-angle) * k + 1))]
      for angle in self.angles]
    self.x_cache = [int(256 * (math.cos(-angle) * k + 1)) * 3
      for angle in self.angles]
  
  def _extract_pixel(self, case):
    x = self.x_cache[case]
    return [v / 256 for v in self.row_refs[case][x:x+3]]
  
  def extract_data(self, case):
    return self._extract_pixel(case) + self._extract_pixel(self.next_cases[case]) + [1]
  
  def wanted(self, case):
    return [abs(case - self.next_cases[case]) / self.cases()]
  
  def cases(self):
    return 120
