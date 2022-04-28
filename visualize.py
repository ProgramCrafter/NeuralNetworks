buffer = [list(' ' * 80) for i in range(25)] + ['_' * 80]

def print_ascii_square(x, y):
  for j in range(x, x + 7):
    buffer[y][j] = '_'
    buffer[y + 3][j] = '_'
  
  for i in range(y + 1, y + 4):
    buffer[i][x] = '|'
    buffer[i][x + 6] = '|'

def print_value(x, y, s):
  for i, c in enumerate(s):
    buffer[y][x + i] = c

def print_sq_with_value(x, y, v):
  print_ascii_square(x, y)
  print_value(x + 2, y + 2, '%.1f' % v)

def flush():
  for row in buffer:
    print(''.join(row))

print_sq_with_value(10, 5, 2.5)
print_sq_with_value(20, 5, 0)
print_sq_with_value(30, 5, 0)

flush()

input()
