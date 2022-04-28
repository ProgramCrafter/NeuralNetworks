import math
import png

class ImageLogger:
  def __init__(self, path):
    mtx = png.Reader(path).read()[2]
    self.image = [list(row) for row in mtx]
  
  def add_mark(self, angle, brightness):
    k = 0.7
    x = int(256 * (math.cos(-angle) * k + 1))
    y = int(256 * (math.sin(-angle) * k + 1))
    
    self.image[y-1][x*3:x*3+3] = [brightness] * 3
    self.image[y+1][x*3:x*3+3] = [brightness] * 3
    self.image[y][x*3-3:x*3+6] = [brightness] * 9
  
  def save(self, path):
    with open(path, 'wb') as f:
      png.Writer(512, 512, greyscale=False).write(
        f, self.image
      )
