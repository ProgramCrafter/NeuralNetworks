# from activators import SigmoidActivator

import traceback
import random
import numba
import math

@numba.njit
def SigmoidActivator__result(v):
  a = math.exp(2 * v)
  return (a - 1.0) / (a + 1.0)
  
  # return 1 / (1 + math.exp(-v))

@numba.njit
def stats(arr):
  n = len(arr)
  
  u = 0.0
  for k in arr: u += k * k
  
  v = 0.0
  for k in arr: v += k
  
  w = 0
  x = 0
  cs = 0
  p = [0, 0, 0, 0]
  for k in arr:
    b1 = 1 if 0.0 <= k else -1
    b2 = 1 if (-0.5 <= k < 0.0 or 0.5 <= k < 1) else -1
    w += b1 + b2
    cs += b1
    x = max(x, abs(cs))
    cs += b2
    x = max(x, abs(cs))
    p[b1 + 1 + (b2 + 1) // 2] += 1
  
  return u / n, v / n, w, x, max(p), p

@numba.njit
def generate_uv():
  u = random.random() * 2.0 - 1.0
  v = random.random() * 2.0 - 1.0
  seed = random.random()
  
  arr = numba.typed.List([0.0] * 20000)
  for j in range(20000):
    arr[j] = SigmoidActivator__result(seed * u)
    seed = SigmoidActivator__result(seed * v)
  
  return (*stats(arr), u, v)

@numba.njit
def generate_uvw():
  u = random.random() * 2.0 - 1.0
  v = random.random() * 2.0 - 1.0
  w = random.random() * 2.0 - 1.0
  seed = random.random()
  
  arr = numba.typed.List([0.0] * 20000)
  for j in range(20000):
    tmp = SigmoidActivator__result(seed * u)
    arr[j] = SigmoidActivator__result(tmp * v)
    seed = SigmoidActivator__result(tmp * w)
  
  return (*stats(arr), u, v, w)

@numba.njit
def generate_uvwxyz():
  u = random.random() * 2.0 - 1.0
  v = random.random() * 2.0 - 1.0
  w = random.random() * 2.0 - 1.0
  x = random.random() * 2.0 - 1.0
  y = random.random() * 2.0 - 1.0
  z = random.random() * 2.0 - 1.0
  seed = random.random()
  
  arr = numba.typed.List([0.0] * 20000)
  for j in range(20000):
    t1 = SigmoidActivator__result(seed * u)
    t2 = SigmoidActivator__result(seed * v)
    arr[j] = SigmoidActivator__result(t1 * w + t2 * x)
    seed = SigmoidActivator__result(t1 * y + t2 * z)
  
  return (*stats(arr), u, v, w, x, y, z)

@numba.njit
def generate_uvwxyzkl():
  u = random.random() * 2.0 - 1.0
  v = random.random() * 2.0 - 1.0
  w = random.random() * 2.0 - 1.0
  x = random.random() * 2.0 - 1.0
  y = random.random() * 2.0 - 1.0
  z = random.random() * 2.0 - 1.0
  k = random.random() * 3.0 - 1.5
  l = random.random() * 3.0 - 1.5
  seed = random.random()
  
  arr = numba.typed.List([0.0] * 20000)
  for j in range(20000):
    t1 = SigmoidActivator__result(seed * u)
    t2 = SigmoidActivator__result(seed * v)
    arr[j] = SigmoidActivator__result(t1 * w + t2 * x + k * seed)
    seed = SigmoidActivator__result(t1 * y + t2 * z + l * seed)
  
  return (*stats(arr), u, v, w, x, y, z, k, l)

@numba.njit
def generate_logistic():
  seed = random.random()
  arr = numba.typed.List([0.0] * 20000)
  
  for j in range(20000):
    rnd = (seed * seed + seed * 2.0) / 3.0
    rnd = (rnd * rnd + rnd * 2.0) / 3.0
    rnd = 4.0 * rnd * (1.0 - rnd)
    rnd = 4.0 * rnd * (1.0 - rnd)
    arr[j] = (rnd * rnd + rnd * 3.0) / 2.0 - 1.0
    seed = 4.0 * seed * (1.0 - seed)
  
  return stats(arr)

def best_stats(stats_arr):
  n = 20000 ** .5
  third = 1/3
  logn = math.log(n)
  key = lambda s: (
      abs(s[0] - third) * 1.5 * 1.0
    + abs(s[1]) * 8.0
    + abs(math.log(s[3]) - logn) * 4.0
    + s[4] / 5000.0 * 2.0)
  return min(stats_arr, key=key)

@numba.njit
def generate_logistic_sequence():
  seed = random.random()
  arr = numba.typed.List([0] * 8388608 * 1024)
  
  for j in range(4194304 * 1024):
    rnd = (seed * seed + seed * 2.0) / 3.0
    rnd = (rnd * rnd + rnd * 2.0) / 3.0
    rnd = 4.0 * rnd * (1.0 - rnd)
    rnd = 4.0 * rnd * (1.0 - rnd)
    rnd = (rnd * rnd + rnd * 3.0) / 2.0 - 1.0
    seed = 4.0 * seed * (1.0 - seed)
    
    arr[j * 2 + 0] = 1 if 0 <= rnd else 0
    arr[j * 2 + 1] = 1 if -0.5 <= rnd < 0 or 0.5 <= rnd < 1 else 0
  
  byt = numba.typed.List([0] * 1048576 * 1024)
  for i in range(1048576 * 1024):
    byt[i] = (arr[i*8+0] * 128 + arr[i*8+1] * 64 + arr[i*8+2] * 32 + arr[i*8+3] * 16
            + arr[i*8+4] * 8   + arr[i*8+5] * 4  + arr[i*8+6] * 2  + arr[i*8+7])
  
  return byt

GENERATE_NETWORKS = False

try:
  with open(__file__ + '/../rand-logistic.bin', 'wb') as f:
    f.write(bytes(generate_logistic_sequence()))
  
  if GENERATE_NETWORKS:
    print('Builtin random:\t%.6f %.6f %d %d %d %s\n' %
      stats(numba.typed.List([random.random() * 2.0 - 1.0 for _ in range(20000)])))
    
    ############################################################################
    
    print('Logistic:\t%.6f %.6f %d %d %d %s\n' % generate_logistic())
    
    ############################################################################
    
    #print('Best U,V:\t%.6f %.6f %d %d %d %s  (u=%.6f\tv=%.6f)\n' %
    #  best_stats(generate_uv() for _ in range(5000)))
    
    ############################################################################
    
    #print('Best U,V,W:\t%.6f %.6f %d %d %d %s  (u=%.6f\tv=%.6f\tw=%.6f)\n' %
    #  best_stats(generate_uvw() for _ in range(5000)))
    
    ############################################################################
    
    #print('Best UVWXYZ:\t%.6f %.6f %d %d %d %s  (%.6f, %.6f, %.6f, %.6f, %.6f, %.6f)\n' %
    #  best_stats(generate_uvwxyz() for _ in range(5000)))
    
    ############################################################################
    
    print('Best UVWXYZKL:\t%.6f %.6f %d %d %d %s  (%.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f)\n' %
      best_stats(generate_uvwxyzkl() for _ in range(5000)))
  else:
    u,v,w,x,y,z = 0.615424, 3.941108, 0.939804, -0.309309, 0.938256, 0.889810
    
    seed = 0.0146
    
    for i in range(100):
      rnd = (seed * seed + seed * 2.0) / 3.0
      rnd = (rnd * rnd + rnd * 2.0) / 3.0
      rnd = 4.0 * rnd * (1.0 - rnd)
      rnd = 4.0 * rnd * (1.0 - rnd)
      seed = 4.0 * seed * (1.0 - seed)
      
      # t1 = SigmoidActivator__result(seed * u)
      # t2 = SigmoidActivator__result(seed * v)
      # rnd  = SigmoidActivator__result(t1 * w + t2 * x)
      # seed = SigmoidActivator__result(t1 * y + t2 * z)
      print('%.6f' % rnd, end=(' ' if (i + 1) % 10 else '\n'))
  
except:
  traceback.print_exc()
finally:
  input('...')
