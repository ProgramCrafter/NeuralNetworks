import math
import json

class IDataSource:
  def __init__(self, path):     pass
  def extract_data(self, case): pass
  def wanted(self, case):       pass
  def cases(self):              pass

class CFProblemTimingsDataSource(IDataSource):
  def __init__(self, path):
    with open(path) as f:
      self.dataset = [
        ([1 / (timing + 1)], [500 / rating])
          for (rating,timing) in json.load(f)
      ]
    self.CASES = len(self.dataset)
  
  def cases(self):
    return self.CASES
  
  def extract_data(self, case):
    return self.dataset[case][0]
  
  def wanted(self, case):
    return self.dataset[case][1]
