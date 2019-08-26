from PIL import Image
import numpy as np
from matplotlib import pylab as plt

for i in range(10000)
    img = np.array(Image.open((str(i+2000)+'.jpg').convert('L'), 'float32')
    np.save('str(i+2000).npy',img)
    print('convert depth image str(i+2000)')
