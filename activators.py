from utils import catch_nan
import math

LReLU_COEF_0 = 0.07
LReLU_COEF_1 = 0.85

class IActivator:
  def result(self, v):     pass
  def derivative(self, v): pass
  def __str__(self):       pass

class LReLUActivator(IActivator):
  @catch_nan
  def result(self, v):
    return max(v * LReLU_COEF_0, v * LReLU_COEF_1)
  
  @catch_nan
  def derivative(self, v):
    return LReLU_COEF_0 if v < 0 else LReLU_COEF_1
  
  def __str__(self):
    return '[LReLU:%.2f:%.2f]\n' % (LReLU_COEF_0, LReLU_COEF_1)

class SigmoidActivator(IActivator):
  @catch_nan
  def result(self, v):
    return 1 / (1 + math.exp(-v))
  
  @catch_nan
  def derivative(self, v):
    w = 1 / (1 + math.exp(-v))
    return w * (1 - w)
  
  def __str__(self):
    return '[Sigmoid]\n'

class TanhActivator(IActivator):
  @catch_nan
  def result(self, v):
    a = math.exp(v)
    b = math.exp(-v)
    return (a - b) / (a + b)
  
  @catch_nan
  def derivative(self, v):
    a = math.exp(v)
    b = math.exp(-v)
    return 1 - (a - b) / (a + b) * (a - b) / (a + b)
  
  def __str__(self):
    return '[tanh]\n'
