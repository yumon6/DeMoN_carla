import csv
import numpy as np

with open('./image/measurements.csv') as f:
    reader = csv.reader(f)
    l = [row for row in reader]
    
for i in range(10000)
    f = open("./image/Rt" + "str(i+2000).txt", "w")
    
    Rx = np.array([[1, 0, 0],
                   [0, np.cos((l[i+103][11])-(l[i+102][11])), np.sin((l[i+103][11])-(l[i+102][11]))],
                   [0, -np.sin((l[i+103][11])-(l[i+102][11])), np.cos((l[i+103][11])-(l[i+102][11]))]])
    Ry = np.array([[np.cos((l[i+103][12])-(l[i+102][12])), 0, -np.sin((l[i+103][12])-(l[i+102][12]))],
                   [0, 1, 0],
                   [np.sin((l[i+103][12])-(l[i+102][12])), 0, np.cos((l[i+103][12])-(l[i+102][12]))]])
    Rz = np.array([[np.cos((l[i+103][13])-(l[i+102][13])), np.sin((l[i+103][13])-(l[i+102][13])), 0],
                   [-np.sin((l[i+103][13])-(l[i+102][13])), np.cos((l[i+103][13])-(l[i+102][13])), 0],
                   [0, 0, 1]])
    R = Rz.dot(Ry).dot(Rx)
    
    r11 = str(R[0][0])
    r12 = str(R[0][1])
    r13 = str(R[0][2])
    r21 = str(R[1][0])
    r22 = str(R[1][1])
    r23 = str(R[1][2])
    r31 = str(R[2][0])
    r32 = str(R[2][1])
    r33 = str(R[2][2])
    t1 = str(10)
    t2 = str(11)
    t3 = str(12)
    f.write(r11 + ' ' + r12 + ' ' + r13 + ' ' + t1 + '\n' + r21 + ' ' + r22 + ' ' + r23 + ' ' + t2 + '\n' + r31 + ' ' + r32 + ' ' + r33 + ' ' + t3 + '\n')
    f.close()
