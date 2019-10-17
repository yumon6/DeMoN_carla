import numpy as np
import math
from PIL import Image
import cv2


for k in range(895):

    img = Image.open('/home/yumon/datasets/new_data/_out/camera_001/depth/000' + str(k+54766) + '.png')
    arr = np.array(img)
    arr = arr.astype(np.float32)
    #print(arr.dtype)
    normalized_depth = np.dot(arr[:, :, :3], [1.0, 256.0, 65536.0])*1000
    normalized_depth /= 16777215.0

    '''
    gray = np.zeros((600,800),dtype='float64')
    for i in range(600):
        for j in range(800):
            R = arr[i][j][0]
            G = arr[i][j][1]
            B = arr[i][j][2]
            gray[i,j] = ((R+(G*256)+(B*256*256))/((256*256*256)-1))*1000
    '''
    normalized_depth = normalized_depth.astype(np.float32)
    #print(normalized_depth.dtype)
    np.save('./data/depth/' + str(k+54766) + '.npy',normalized_depth)
    print('created' + str(k+54766) + '.npy')
~                                              
