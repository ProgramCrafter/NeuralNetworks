import math
import json

class IDataSource:
  def __init__(self, path):     pass
  def extract_data(self, case): pass
  def wanted(self, case):       pass
  def cases(self):              pass

class CFProblemTimingsDataSource(IDataSource):
  def __init__(self, path):
    self.dataset = []
    
    with open(path) as f:
      for (rating, lang, time, memory) in json.load(f):
        lang_const = 1
        
        if 'Python' in lang:
          lang_const = 0.5
        elif 'PyPy' in lang or 'Java' in lang:
          lang_const = 0.8
        
        self.dataset.append(
          (600 / rating, lang_const, 1 / (time + 1), 1 / (memory / 2**20 + 1))
        )
    
    self.CASES = len(self.dataset)
  
  def cases(self):
    return self.CASES
  
  def extract_data(self, case):
    return self.dataset[case][1:]
  
  def wanted(self, case):
    return self.dataset[case][:1]
