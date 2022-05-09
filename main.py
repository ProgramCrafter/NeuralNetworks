import traceback
import itertools
import cProfile
import random
import math

TRAIN_SPEED = 8e-5
TRAIN_LIMIT = 20
TRAIN_BLIMIT = 7e-3
COEF_LIMIT = 9999

from data_source import IconDataExtractor
from activators import TanhActivator
from utils import catch_nan

class InitialWeightsGenerator:
  INIT_WEIGHTS = None
  
  def generate(self, iterable):
    if not self.INIT_WEIGHTS:
      return [random.random() * 2 - 1 for it in iterable]
    else:
      return [random.random() * 0 + self.INIT_WEIGHTS.__next__() for it in iterable]

class INeuron:
  def __init__(self, activator, initializer, previous_layer): pass
  def calculate(self):       pass
  def sprintf_weights(self): pass

class InputValue(INeuron):
  def __init__(self, activator, initializer, previous_layer):
    self.activator = activator
    self.coef = initializer.generate(' ')[0]
    self.value = 0
    self.cache = None
    self.coefs = [self.coef]
    self.next_layer = []
  
  @catch_nan
  def calculate(self):
    if not self.cache:
      self.cache = self.activator.result(self.value)
    return self.cache
  
  @catch_nan
  def delta_as_last(self, error):
    s = self.coef * self.value
    d = self.activator.derivative(s)
    
    return d * error
  
  @catch_nan
  def delta_as_not_last(self, next_deltas):
    s = self.coef * self.value
    d = self.activator.derivative(s)
    
    mult = 0
    for i, neuron in enumerate(self.next_layer):
      mult += neuron[0].coefs[neuron[1]] * next_deltas[i]
    
    return d * mult
  
  @catch_nan
  def delta(self, next_deltas):
    if self.next_layer:
      a = self.delta_as_not_last(next_deltas)
    else:
      a = self.delta_as_last(next_deltas)
    
    train_value = TRAIN_SPEED * a * self.value
    
    if train_value < -TRAIN_LIMIT: train_value = -TRAIN_LIMIT
    elif train_value > TRAIN_LIMIT: train_value = TRAIN_LIMIT
    
    k = self.coef - train_value
    if k < -COEF_LIMIT: k = -COEF_LIMIT
    elif k > COEF_LIMIT: k = COEF_LIMIT
    self.coef = k
    self.coefs = [k]
    self.cache = None
    
    return a

class Neuron(INeuron):
  def __init__(self, activator, initializer, previous_layer):
    self.activator = activator
    self.previous_layer = previous_layer
    self.next_layer = []
    self.coefs = initializer.generate(previous_layer)
    self.cache = None
    
    for i,p in enumerate(self.previous_layer):
      p.next_layer.append((self, i))
  
  @catch_nan
  def calculate(self):
    if not self.cache:
      s = sum(self.coefs[i] * v.calculate() for i, v in enumerate(self.previous_layer))
      self.cache = self.activator.result(s)
    
    return self.cache
  
  @catch_nan
  def delta_as_last(self, error):
    s = sum(self.coefs[i] * v.calculate() for i, v in enumerate(self.previous_layer))
    
    d = self.activator.derivative(s)
    
    return d * error
  
  @catch_nan
  def delta_as_not_last(self, next_deltas):
    s = sum(self.coefs[i] * v.calculate() for i, v in enumerate(self.previous_layer))
    
    d = self.activator.derivative(s)
    
    mult = 0
    for i, neuron in enumerate(self.next_layer):
      mult += neuron[0].coefs[neuron[1]] * next_deltas[i]
    
    return d * mult
  
  @catch_nan
  def delta(self, next_deltas):
    if self.next_layer:
      a = self.delta_as_not_last(next_deltas)
    else:
      a = self.delta_as_last(next_deltas)
    
    for i, prev_neuron in enumerate(self.previous_layer):
      train_value = TRAIN_SPEED * a * prev_neuron.calculate()
      
      if train_value < -TRAIN_LIMIT: train_value = -TRAIN_LIMIT
      elif train_value > TRAIN_LIMIT: train_value = TRAIN_LIMIT
      
      k = self.coefs[i] - train_value
      if k < -COEF_LIMIT: k = -COEF_LIMIT
      elif k > COEF_LIMIT: k = COEF_LIMIT
      self.coefs[i] = k
    
    self.cache = None
    
    return a

class SparelinkNeuralNetwork:
  def __init__(self, activator, initializer, levels):
    self.activator_prefix = str(activator)
    
    self.layers = [[InputValue(activator, initializer, None) for i in range(4 ** levels * 3)]]
    
    # first layer (4 ** levels == 256) - neurons with 3 inputs (R, G, B)
    self.layers.append(
      [Neuron(activator, initializer, self.layers[0][i*3:i*3+3]) for i in range(4 ** levels)]
    )
    
    # second layer (4 ** 3 == 64)
    # third layer  (4 ** 2 == 16)
    # fourth layer (4 ** 1 == 4)
    # final layer  (4 ** 0 == 1)
    for i in range(levels)[::-1]:
      last_layer = self.layers[-1]
      size = 4 ** i
      
      self.layers.append([Neuron(activator, initializer, [
        last_layer[k], last_layer[k + size], last_layer[k + 2 * size], last_layer[k + 3 * size]
      ]) for k in range(size)])
    
  def set_inputs(self, input_values):
    for i, neuron in enumerate(self.layers[0]):
      neuron.value = input_values[i]
    
    for layer in self.layers:
      for neuron in layer:
        neuron.cache = None
  
  def calculate(self):
    for neuron in self.layers[-1]:
      yield neuron.calculate()
  
  def train(self, wanted):
    # back errors propagation
    output = list(self.calculate())
    
    # output layer
    deltas = [
      neuron.delta(output[i] - wanted[i])
          for i, neuron in enumerate(self.layers[-1])
    ]
    
    for layer in self.layers[:-1][::-1]:
      deltas = [neuron.delta(deltas) for neuron in layer]
  
  def enum_weights(self):
    for layer in self.layers:
      for neuron in layer:
        yield from neuron.coefs

def epoch(net, data):
  sum_sq = 0
  
  cases = list(range(data.cases()))
  random.shuffle(cases)
  
  trains = data.cases() // 1
  
  for case in cases:
    net.set_inputs(data.extract_data(case))
    
    net_result = net.calculate()
    wanted_result = data.wanted(case)
    
    for i, nr in enumerate(net_result):
      sum_sq += (nr - wanted_result[i]) * (nr - wanted_result[i])
    
    if trains > 0:
      net.train(wanted_result)
      trains -= 1
    
  return sum_sq / data.cases()

def main():
  try:
    random.seed(0x14609A25)
    
    net = SparelinkNeuralNetwork(TanhActivator(), InitialWeightsGenerator(), 4)
    data = IconDataExtractor(__file__ + '/../icons/r-',
      ['oo-0-0.png', 'tg-0-0.png', 'ya-0-0.png'])
    
    print(net, data)
    last_distance = epoch(net, data)
    
    try:
      for i in range(100001):
        cur_distance = epoch(net, data)
        
        if i % 200 == 0:
          print('Epoch %6d - square distance = %.4f (delta = %.4f)' % (i, cur_distance, cur_distance - last_distance))
          last_distance = min(last_distance, cur_distance)
        
        if cur_distance < 0.005:
          break
    except KeyboardInterrupt:
      pass
    except Exception:
      raise
    
    print('\nResults:')
    
    sum_distance = 0
    sum_abs = 0
    
    for case in range(data.cases()):
      net.set_inputs(data.extract_data(case))
      
      net_result = list(net.calculate())
      wanted_result = data.wanted(case)
      
      print('Net output = %.4f; wanted = %d' % (net_result[0], wanted_result[0]))
      
      for i, nr in enumerate(net_result):
        sum_distance += (nr - wanted_result[i]) * (nr - wanted_result[i])
        sum_abs += abs(nr - wanted_result[i])
    
    print('\nMedian square error: %.4f' % (sum_distance / data.cases()))
    print('Median absolute error: %.4f' % (sum_abs / data.cases()))
    
    print('\nWeights:')
    print([int(w * 1000) / 1000 for w in net.enum_weights()])
  except:
    traceback.print_exc()

# cProfile.run('main()', sort='time')
main()
input()
