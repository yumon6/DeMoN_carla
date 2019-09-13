from PIL import Image, ImageOps
import numpy as np
from matplotlib import pylab as plt


for k in range(10000):
    img = np.array(Image.open('./image/depth_logarithmic_gray/' + str(k+3000) + '.jpg').convert('I'), 'float32')
    def calc_double(n):
        return n + 40

    img2 = list(map(calc_double, img))

    for i in range(600):
        for j in range(800):
            if img2[i][j] > 255:
                img2[i][j] = 255

    np.save('./image/depth_gray/' + str(k+3000) + '.npy',img2)
    print(str(k+3000) + '.jpg saved')
