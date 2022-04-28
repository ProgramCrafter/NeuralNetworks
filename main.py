import traceback
import itertools
import cProfile
import random
import math

TRAIN_SPEED = 8e-5
TRAIN_LIMIT = 20
TRAIN_BLIMIT = 7e-3
COEF_LIMIT = 9999

from data_source import HsvDataExtractor
from image_logger import ImageLogger
from activators import TanhActivator
from utils import catch_nan

weights = [0.348, -0.096, 1.191, 0.723,
  0.374,-0.911,-0.584,-0.292,  0.271,0.878,-1.362,-0.736,  0.488,-0.007,0.825,0.967,
  -0.578,0.995,0.896
].__iter__()

class InitialWeightsGenerator:
  def generate(self, iterable):
    return [(random.random() * 2 - 1) * 0 + weights.__next__() for it in iterable]

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
  
  @catch_nan
  def calculate(self):
    if not self.cache:
      self.cache = self.activator.result(self.value)
    return self.cache
  
  def sprintf_weights(self):
    return ('%.3f' % self.coef).ljust(16) # + ' (% 8.3f)' % self.calculate()
  
  @catch_nan
  def delta_as_last(self, error):
    s = self.coef * self.value
    d = self.activator.derivative(s)
    
    return d * error
  
  @catch_nan
  def delta_as_not_last(self, next_deltas, self_index):
    s = self.coef * self.value
    d = self.activator.derivative(s)
    
    mult = 0
    for i, neuron in enumerate(self.next_layer):
      mult += neuron.coefs[self_index] * next_deltas[i]
    
    return d * mult
  
  @catch_nan
  def delta(self, next_deltas, self_index):
    if self.next_layer:
      a = self.delta_as_not_last(next_deltas, self_index)
    else:
      a = self.delta_as_last(next_deltas)
    
    train_value = TRAIN_SPEED * a * self.value
    
    if train_value < -TRAIN_LIMIT: train_value = -TRAIN_LIMIT
    elif train_value > TRAIN_LIMIT: train_value = TRAIN_LIMIT
    
    k = self.coef - train_value
    if k < -COEF_LIMIT: k = -COEF_LIMIT
    elif k > COEF_LIMIT: k = COEF_LIMIT
    self.coef = k
    
    return a
  
  def uncache(self):
    self.cache = None

class Neuron(INeuron):
  def __init__(self, activator, initializer, previous_layer):
    self.activator = activator
    self.previous_layer = previous_layer
    self.next_layer = None
    self.coefs = initializer.generate(previous_layer)
    self.cache = None
  
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
  def delta_as_not_last(self, next_deltas, self_index):
    s = sum(self.coefs[i] * v.calculate() for i, v in enumerate(self.previous_layer))
    
    d = self.activator.derivative(s)
    
    mult = 0
    for i, neuron in enumerate(self.next_layer):
      mult += neuron.coefs[self_index] * next_deltas[i]
    
    return d * mult
  
  @catch_nan
  def delta(self, next_deltas, self_index):
    if self.next_layer:
      a = self.delta_as_not_last(next_deltas, self_index)
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
  
  def sprintf_weights(self):
    return ','.join('%.3f' % v for v in self.coefs)#.ljust(26) #+ ' (% 8.3f)' % self.calculate()
  
  def uncache(self):
    self.cache = None
    
    for neuron in self.previous_layer: neuron.uncache()

class NeuralNetwork:
  def __init__(self, activator, initializer, inputs, layer_sizes):
    self.layers = [[InputValue(activator, initializer, None) for i in range(inputs)]]
    
    self.activator_prefix = str(activator)
    
    for size in layer_sizes:
      self.layers.append([Neuron(activator, initializer, self.layers[-1]) for i in range(size)])
      for neuron in self.layers[-2]:
        neuron.next_layer = self.layers[-1]
    
  def set_inputs(self, input_values):
    for i, neuron in enumerate(self.layers[0]):
      neuron.value = input_values[i]
    
    for layer in self.layers:
      for neuron in layer:
        neuron.cache = None
    
    # for neuron in self.layers[-1]:
    #   neuron.uncache()
  
  def calculate(self):
    for neuron in self.layers[-1]:
      yield neuron.calculate()
  
  def sprintf_weights(self):
    return self.activator_prefix + '\n'.join( # layers
      ' | '.join(
        neuron.sprintf_weights()
      for neuron in layer)
    for layer in self.layers)
  
  def train(self, wanted):
    # back errors propagation
    output = list(self.calculate())
    
    # output layer
    deltas = [
      neuron.delta(output[i] - wanted[i], i)
          for i, neuron in enumerate(self.layers[-1])
    ]
    
    for layer in self.layers[:-1][::-1]:
      deltas = [neuron.delta(deltas, i) for i, neuron in enumerate(layer)]

def epoch(net, data):
  sum_sq = 0
  
  cases = list(range(data.cases()))
  random.shuffle(cases)
  
  for case in cases:
    net.set_inputs(data.extract_data(case))
    
    net_result = net.calculate()
    wanted_result = data.wanted(case)
    # net_result = wanted_result
    
    for i, nr in enumerate(net_result):
      sum_sq += (nr - wanted_result[i]) * (nr - wanted_result[i])
    
    net.train(wanted_result)
    
  return sum_sq / data.cases()

def main():
  try:
    random.seed(0x14609A25)
    
    net = NeuralNetwork(TanhActivator(), InitialWeightsGenerator(), 4, [3, 1])
    data = HsvDataExtractor(__file__ + '/../hsv.png')
    logger = ImageLogger(__file__ + '/../hsv.png')
    
    print(net, data)
    last_distance = epoch(net, data)
    
    try:
      for i in range(50001):
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
    sum_angle = 0
    d = []
    for case in range(data.cases()):
      net.set_inputs(data.extract_data(case))
      
      net_result = list(net.calculate())
      wanted_result = data.wanted(case)
      # net_result = wanted_result
      
      for i, nr in enumerate(net_result):
        sum_distance += (nr - wanted_result[i]) ** 2
      
      logger.add_mark(net_result[0] * 2 * math.pi, (case * 256) // data.cases())
      sum_angle += abs(net_result[0] - wanted_result[0])
      
      wr = wanted_result[0] * 360
      nr = net_result[0] * 360
      d.append(int(abs(wr - nr)))
      print('%d:%d (%d)' % (wr, nr, abs(wr - nr)), end='\t')
      if (case + 1) % 7 == 0: print()
    
    d.sort()
    print('\n\nDistances array:')
    for chunk in range(0, len(d), 30): print(*d[chunk:chunk+30])
    
    logger.save(__file__ + '/../nn-output.png')
    
    print('\nMedian distance: %.4f' % (sum_distance / data.cases()))
    print('Median angle: %.4f' % (sum_angle / data.cases() * 360))
    
    print('\nWeights:')
    print(net.sprintf_weights())
  except:
    traceback.print_exc()

# cProfile.run('main()', sort='time')
main()
input()