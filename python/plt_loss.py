import csv

import matplotlib.pyplot as plt

csv_path = '230512loss2_art3060'

rows = []
with open(csv_path + ".csv") as f:
    reader = csv.reader(f)
    rows = [row for row in reader]

#data = [float(x) for x in rows[0][1:-1]]
#plt.plot(data)
"""
data1 = [float(x) for x in rows[0][0:59]]
data2 = [float(x) for x in rows[0][60:119]]
data3 = [float(x) for x in rows[0][240:299]]

plt.plot(data1)
plt.plot(data2)
plt.plot(data3)
"""

plt.savefig(csv_path + ".png")
plt.show()
