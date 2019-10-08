from PIL import Image
from matplotlib import pylab as plt
import numpy as np
import cv2
from pylab import *
import csv


with open('../episode1/measurements.csv') as f:
    reader = csv.reader(f)
    l = [row for row in reader]

for i in range(850):
    f = open("../episode1/Rt/" + str(i+15000) + ".txt", "w")
    #f = open("../episode1/Rt/" + str(i+15697) + ".txt", "w") 

    x = 500 - float(l[i+13103][5])*13
    y = 500 - float(l[i+13103][6])*10

    round(y)
    round(x)
    #print(x)

    img = cv2.imread('../episode1/rgb/' + str(i+15000) + '.jpg')
    cv2.arrowedLine(img, (200, 500), (200, int(x)), (0, 0, 255), thickness=3)
    cv2.arrowedLine(img, (600, 500), (600, int(y)), (0, 255, 0), thickness=3)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.tick_params(labelbottom=False,
                labelleft=False,
                labelright=False,
                labeltop=False)
    plt.tick_params(bottom=False,
                left=False,
                right=False,
                top=False)

    plt.imshow(img)
    plt.savefig('./image2/' + str(i+15000) + '.jpg')
    print(str(i))
    #plt.show()
    clf()
