import csv
import numpy as np

with open('/home/yumon/datasets/new_data/_out/measurement.csv') as f:
    reader = csv.reader(f)
    l = [row for row in reader]

s = -54764 + 54766

for i in range(894):
    f = open("./data/Rt/" + str(i+54766) + ".txt", "w")
    #print(str(i+54766))
    #s = -54764 + 54766
    print(i+54766)
    #print(l[i+s][10])
    if i == 0:
        r = p = y = 0
    else:
        r1 = float(l[i+s][8])
        r2 = float(l[i+s-1][8])
        p1 = float(l[i+s][9])
        p2 = float(l[i+s-1][9])
        y1 = float(l[i+s][10])
        y2 = float(l[i+s-1][10])
        if r1>0 and r2<0:
            r = -(180 - r1) - (180 + r2)
        elif r1<0 and r2>0:
            r = (180 + r1) + (180 - r2)
        else:
            r = r1 - r2
        p = p1 - p2
        y = y1 - y2
    '''
    else:    
        r = (float(l[i+s][8]))-(float(l[i+s-1][8]))   
        p = (float(l[i+s][9]))-(float(l[i+s-1][9]))
        y = (float(l[i+s][10]))-(float(l[i+s-1][10]))
    '''

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(r), np.sin(r)],
                   [0, -np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, -np.sin(p)],
                   [0, 1, 0],
                   [np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), np.sin(y), 0],
                   [-np.sin(y), np.cos(y), 0],
                   [0, 0, 1]])
    R = Rz.dot(Ry).dot(Rx)
    #print(y)
    if i == 0:
        Tx = Ty = Tz = 0
    else:
        Tx = float(l[i+s][15])-float(l[i+s-1][15])
        Ty = float(l[i+s][16])-float(l[i+s-1][16])
        Tz = float(l[i+s][17])-float(l[i+s-1][17])
    #print(Tx)

    r11 = str(R[0][0])
    r12 = str(R[0][1])
    r13 = str(R[0][2])
    r21 = str(R[1][0])
    r22 = str(R[1][1])
    r23 = str(R[1][2])
    r31 = str(R[2][0])
    r32 = str(R[2][1])
    r33 = str(R[2][2])
    t1 = str(Tx)
    t2 = str(Ty)
    t3 = str(Tz)
    print(t1 + ' ' + t2 + ' ' + t3)
    f.write(' '.join([r11,r12,r13,t1]) + '\n' + ' '.join([r21,r22,r23,t2]) + '\n' + ' '.join([r31,r32,r33,t3]))
    f.close()
