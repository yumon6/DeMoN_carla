import csv

with open('./image/measurements.csv') as f:
    reader = csv.reader(f)
    l = [row for row in reader]
    
for i in range(10000)
    f = open("./image/Rt" + "str(i+2000).txt", "w")
    r11 = str(l[])
    r12 = str(2)
    r13 = str(3)
    r21 = str(4)
    r22 = str(5)
    r23 = str(6)
    r31 = str(7)
    r32 = str(8)
    r33 = str(9)
    t1 = str(10)
    t2 = str(11)
    t3 = str(12)
    f.write(r11 + ' ' + r12 + ' ' + r13 + ' ' + t1 + '\n' + r21 + ' ' + r22 + ' ' + r23 + ' ' + t2 + '\n' + r31 + ' ' + r32 + ' ' + r33 + ' ' + t3 + '\n')
    f.close()
