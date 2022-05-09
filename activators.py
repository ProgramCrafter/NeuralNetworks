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
    a = math.exp(2 * v)
    return (a - 1.0) / (a + 1.0)
  
  def slow_result(self, v):
    a = math.exp(v)
    b = math.exp(-v)
    return (a - b) / (a + b)
  
  @catch_nan
  def derivative(self, v):
    a = math.exp(2 * v)
    b = math.exp(4 * v)
    return 4.0 * a / (b + 2.0 * a + 1)
  
  def slow_derivative(self, v):
    a = math.exp(v)
    b = math.exp(-v)
    return 1 - (a - b) / (a + b) * (a - b) / (a + b)
  
  def __str__(self):
    return '[tanh]\n'

if __name__ == '__main__':
  import timeit
  
  init = 't = TanhActivator()'
  glob = {'TanhActivator': TanhActivator}
  
  print('TanhActivator speed')
  
  print(end='- result()\t')
  print(*['%.4f' % v
    for v in sorted(timeit.repeat('t.result(0.5)', init, globals=glob, repeat=12))])
  
  print(end='- slow_result()\t')
  print(*['%.4f' % v
    for v in sorted(timeit.repeat('t.slow_result(0.5)', init, globals=glob, repeat=12))])
  
  print(end='\n- derivative()\t')
  print(*['%.4f' % v
    for v in sorted(timeit.repeat('t.derivative(0.5)', init, globals=glob, repeat=12))])
  
  print(end='- slow_deriv()\t')
  print(*['%.4f' % v
    for v in sorted(timeit.repeat('t.slow_derivative(0.5)', init, globals=glob, repeat=12))])
  
  input('\n...')
