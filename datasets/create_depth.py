from PIL import Image
import numpy as np
from matplotlib import pylab as plt

for i in range(10000):
    img = np.array(Image.open('./image/depth_logarithmic_gray/' + str(i+2000)+'.jpg').convert('L'), 'float32')
    np.save('./image/depth/' + str(i+2000) + '.npy',img)
    print('Convert ' + str(i+2000) + '.jpg to ' + str(i+2000) + '.npy')
