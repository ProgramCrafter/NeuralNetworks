import math
import png

class IDataSource:
  def __init__(self, path):     pass
  def extract_data(self, case): pass
  def wanted(self, case):       pass
  def cases(self):              pass

class HsvDataExtractor(IDataSource):
  CASES = 120
  
  def __init__(self, path):
    mtx = png.Reader(path).read()[2]
    self.image = [list(row) for row in mtx]
    
    k = 0.7
    self.angles = [case * (360 // self.CASES) * math.pi / 180
      for case in range(self.cases())]
    self.row_refs = [self.image[int(256 * (math.sin(-angle) * k + 1))]
      for angle in self.angles]
    self.x_cache = [int(256 * (math.cos(-angle) * k + 1)) * 3
      for angle in self.angles]
  
  def extract_data(self, case):
    # angle = case * (360 // self.CASES) * math.pi / 180
    # x = int(256 * (math.cos(-angle) * k + 1))
    # y = int(256 * (math.sin(-angle) * k + 1))
    
    # return (*[v/256 for v in self.image[y][x*3:x*3+3]], 1)
    
    x = self.x_cache[case]
    return [v / 256 for v in self.row_refs[case][x:x+3]] + [1]
  
  def wanted(self, case):
    return [case / 120, 0.7]
  
  def cases(self):
    return 120
