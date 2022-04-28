SKIP_CATCH_NAN = True

def catch_nan(f):
  if SKIP_CATCH_NAN:
    return f
  
  def inner(*a, **k):
    r = f(*a, **k)
    
    if r != r:
      raise Exception('%s returned NaN' % f.__qualname__)
    return r
  
  inner.__name__ = f.__name__
  inner.__qualname__ = f.__qualname__
  
  return inner
